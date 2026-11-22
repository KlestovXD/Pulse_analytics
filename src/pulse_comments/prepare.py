import csv
import io
import json
import re
from collections import Counter
from dataclasses import dataclass

from src.pulse_quality.report import write_text

from .schema import REVIEW_FIELDS, SCHEMA, json_loads, sha256, validate_label


@dataclass
class Inventory:
    texts: dict
    links: list
    checksums: dict


def read_csv(path, required):
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), delimiter=";", strict=True)
    fields = reader.fieldnames or []
    if not set(required) <= set(fields) or len(fields) != len(set(fields)):
        raise ValueError(f"{path.name}: invalid header")
    rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"{path.name}: malformed row")
    return rows, sha256(data)


def index_by_id(rows, field):
    indexed = {row[field]: row for row in rows}
    if len(indexed) != len(rows) or "" in indexed:
        raise ValueError(f"Empty or repeated {field}")
    return indexed


def content_status(text):
    trimmed = text.strip()
    if not trimmed:
        return "empty"
    if re.fullmatch(r"[\W_]+|нет комментариев[.!]?|без комментариев[.!]?|нет[.!]?|n/a", trimmed, flags=re.I):
        return "placeholder"
    return "substantive"


def build_inventory(processed, raw, employees):
    common = ("comment_id", "employee_id", "wave", "team_id", "division", "text")
    original, raw_sha = read_csv(raw, (*common, "submitted_at"))
    derived, processed_sha = read_csv(processed, (*common, "source_record", "submitted_at_raw", "quality_rules",
                                                   "usable_team_text"))
    roster, employee_sha = read_csv(employees, ("employee_id",))
    known = index_by_id([{"employee_id": row["employee_id"].strip()} for row in roster], "employee_id")
    source = index_by_id([{"comment_id": row["comment_id"].strip()} for row in original], "comment_id")
    target = index_by_id(derived, "comment_id")
    if source.keys() != target.keys():
        raise ValueError("Processed comments do not cover exactly the raw IDs")

    texts, links = {}, []
    for record, row in enumerate(original, 1):
        identifier = row["comment_id"].strip()
        cleaned = target[identifier]
        structured_matches = all(row[field].strip() == cleaned[field] for field in common if field != "text")
        if (not structured_matches or row["text"] != cleaned["text"]
                or row["submitted_at"].strip() != cleaned["submitted_at_raw"]
                or cleaned["source_record"] != str(record)):
            raise ValueError(f"Raw and processed comments disagree at record {record}")

        text = row["text"]
        key = sha256(text.encode("utf-8"))
        status = content_status(text)
        item = texts.setdefault(key, {"text_id": key, "text": text, "content_status": status, "occurrences": 0})
        if item["text"] != text:
            raise ValueError("Text fingerprint collision")
        item["occurrences"] += 1
        author = cleaned["employee_id"]
        author_status = "missing_author" if not author else ("confirmed" if author in known else "unknown_author")
        rules = set(filter(None, cleaned["quality_rules"].split("|")))
        if status == "placeholder":
            rules.add("comment_placeholder")
        if status == "empty":
            rules.add("comment_empty")
        if cleaned["usable_team_text"] not in {"True", "False"}:
            raise ValueError(f"Invalid usability flag at record {record}")
        eligible = (
            cleaned["usable_team_text"] == "True" and author_status == "confirmed" and status == "substantive"
        )
        links.append({
            "comment_id": identifier, "source_record": record, "text_id": key, "wave": cleaned["wave"],
            "employee_id": author, "team_id": cleaned["team_id"], "division": cleaned["division"],
            "content_status": status, "author_status": author_status, "quality_rules": "|".join(sorted(rules)),
            "usable_team_text": eligible,
        })
    return Inventory(dict(sorted(texts.items())), links, {
        "raw_comments": raw_sha, "processed_comments": processed_sha, "raw_employees": employee_sha,
    })


def select_review_ids(inventory, sample_path, *, size=30):
    ids = json_loads(sample_path.read_text(encoding="utf-8"))
    if (not isinstance(ids, list) or not ids or (size is not None and len(ids) != size)
            or any(not isinstance(key, str) for key in ids)):
        raise ValueError("Expected a nonempty list of text fingerprints" + (f" with size {size}" if size else ""))
    if len(set(ids)) != len(ids) or any(key not in inventory.texts for key in ids):
        raise ValueError("Review IDs must identify distinct texts in this inventory")
    if any(inventory.texts[key]["content_status"] == "empty" for key in ids):
        raise ValueError("Select a written placeholder rather than an empty cell for the manual text sample")
    return ids


def write_csv(path, fields, rows, *, bom=False):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=";", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, ("\ufeff" if bom else "") + buffer.getvalue())


TEXT_COLUMNS = ("text_id", "text", "content_status", "occurrences")
LINK_COLUMNS = ("comment_id", "source_record", "text_id", "wave", "employee_id", "team_id", "division",
                "content_status", "author_status", "quality_rules", "usable_team_text")
REVIEW_COLUMNS = ("text_id", "text", "gold_status", *(f"gold_{field}" for field in REVIEW_FIELDS), "reviewer_note")
REVIEW_TEMPLATE = "manual_review_template_v2.csv"


def export_inventory(inventory, output, sample_ids):
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "unique_texts.csv", TEXT_COLUMNS, inventory.texts.values())
    write_csv(output / "comment_text_map.csv", LINK_COLUMNS, inventory.links)
    if sample_ids is not None:
        rows = ({"text_id": key, "text": inventory.texts[key]["text"]} for key in sample_ids)
        write_csv(output / REVIEW_TEMPLATE, REVIEW_COLUMNS, rows, bom=True)
    metadata = {
        "input_sha256": inventory.checksums,
        "comments": len(inventory.links),
        "unique_texts": len(inventory.texts),
        "unique_by_content_status": dict(sorted(Counter(r["content_status"] for r in inventory.texts.values()).items())),
        "rows_by_content_status": dict(sorted(Counter(r["content_status"] for r in inventory.links).items())),
        "rows_by_author_status": dict(sorted(Counter(r["author_status"] for r in inventory.links).items())),
        "review_sample": len(sample_ids) if sample_ids is not None else None,
    }
    write_text(output / "inventory.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def validate_review(path, inventory, sample_ids):
    rows, _ = read_csv(path, REVIEW_COLUMNS)
    indexed = index_by_id(rows, "text_id")
    if indexed.keys() != set(sample_ids):
        raise ValueError("Manual file must cover the exact frozen review sample")
    invalid, reference = [], {}
    for index, row in enumerate(rows, 1):
        key = row["text_id"]
        text = inventory.texts[key]["text"]
        if text != row["text"]:
            invalid.append(f"row {index}: text has changed")
            continue
        try:
            status = row["gold_status"].strip()
            values = {name: row[f"gold_{name}"] for name in REVIEW_FIELDS}
            if status == "placeholder":
                if any(value.strip() for value in values.values()):
                    raise ValueError("gold_* categories and evidence must be empty for a placeholder")
                label = None
            elif status == "substantive":
                label = {
                    name: [item.strip() for item in value.split("|")]
                    if SCHEMA["properties"][name]["type"] == "array" else value
                    for name, value in values.items()
                }
                validate_label(label, text, manual=True)
            else:
                raise ValueError("gold_status: expected substantive or placeholder")
            reference[key] = {"status": status, "label": label}
        except ValueError as exc:
            invalid.append(f"row {index}: {exc}")
    if invalid:
        raise ValueError(f"{len(invalid)} review rows need attention. " + "; ".join(invalid[:4]))
    return reference
