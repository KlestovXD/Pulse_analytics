import json
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime

from src.pulse_quality.report import write_text

from .gemini import MODEL_ID, GeminiClient, GeminiError, InvalidGeminiResponse
from .schema import INSTRUCTION_HASH, INSTRUCTION_VERSION, json_loads, validate_label


def json_lines(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)


def export_labels(inventory, output, labels, manual_reviews):
    records = []
    for link in inventory.links:
        entry = labels.get(link["text_id"])
        review = manual_reviews.get(link["text_id"])
        state = link["content_status"]
        if state == "substantive":
            state = "classified" if entry is not None else "pending"
        record = link | {"label_status": state, "annotation": entry}
        if state == "pending" and review is not None:
            record.update(label_status="needs_manual_review", manual_review=review)
        records.append(record)
    write_text(output / "labels_by_text.jsonl", json_lines(labels[key] for key in sorted(labels)))
    write_text(output / "comment_labels.jsonl", json_lines(records))
    write_text(output / "manual_review_by_text.jsonl",
               json_lines(manual_reviews[key] for key in sorted(manual_reviews)))


def record_manual_review(db, key, error):
    review = {
        "text_id": key, "model": MODEL_ID, "instruction_version": INSTRUCTION_VERSION,
        "instruction_sha256": INSTRUCTION_HASH, "diagnostics": error.diagnostics,
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with db:
        db.execute("INSERT INTO manual_reviews VALUES (?, ?, ?, ?)", (
            key, MODEL_ID, INSTRUCTION_HASH, json.dumps(review, ensure_ascii=False, allow_nan=False),
        ))
    print(f"Needs manual review {key}: {error}", file=sys.stderr)
    return review


def run_classification(inventory, output, rpm, max_texts, *, client_factory=GeminiClient, cache=None, provider=None):
    output.mkdir(parents=True, exist_ok=True)
    database = cache if cache is not None else output / "gemini_cache.sqlite3"
    with closing(sqlite3.connect(database)) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS labels (
            text_id TEXT NOT NULL, model TEXT NOT NULL, instruction_hash TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY (text_id, model, instruction_hash)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS manual_reviews (
            text_id TEXT NOT NULL, model TEXT NOT NULL, instruction_hash TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY (text_id, model, instruction_hash)
        )""")
        labels = {
            key: json_loads(payload) for key, payload in db.execute(
                "SELECT text_id, payload FROM labels WHERE model = ? AND instruction_hash = ?",
                (MODEL_ID, INSTRUCTION_HASH),
            ) if key in inventory.texts
        }
        for key, annotation in labels.items():
            if (annotation.get("text_id") != key or annotation.get("model") != MODEL_ID
                    or annotation.get("instruction_sha256") != INSTRUCTION_HASH
                    or provider is not None and annotation.get("provider") != provider):
                raise ValueError("Cached label provenance does not match its key")
            validate_label(annotation["label"], inventory.texts[key]["text"])
        manual_reviews = {
            key: json_loads(payload) for key, payload in db.execute(
                "SELECT text_id, payload FROM manual_reviews WHERE model = ? AND instruction_hash = ?",
                (MODEL_ID, INSTRUCTION_HASH),
            ) if key in inventory.texts and key not in labels
        }
        pending = [key for key, row in inventory.texts.items()
                   if row["content_status"] == "substantive" and key not in labels and key not in manual_reviews]
        try:
            client = client_factory(rpm) if pending else None
            for key in pending[:max_texts]:
                try:
                    annotation = client.classify(inventory.texts[key]["text"]) | {"text_id": key}
                except InvalidGeminiResponse as exc:
                    manual_reviews[key] = record_manual_review(db, key, exc)
                    continue
                except GeminiError as exc:
                    raise GeminiError(f"{exc} Pending text ID {key}.") from None
                if provider is not None and annotation.get("provider") != provider:
                    raise ValueError("Unexpected source for a new annotation")
                with db:
                    db.execute("INSERT INTO labels VALUES (?, ?, ?, ?)", (
                        key, MODEL_ID, INSTRUCTION_HASH, json.dumps(annotation, ensure_ascii=False, allow_nan=False),
                    ))
                labels[key] = annotation
        finally:
            export_labels(inventory, output, labels, manual_reviews)
    return len(labels), len(manual_reviews), max(0, len(pending) - max_texts)
