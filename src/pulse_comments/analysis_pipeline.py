import json
from collections import Counter

from src.pulse_quality.report import write_text

from .analytics import TOPIC_TO_SCORE, descriptive_tables, scored_comments, topic_score_table, turnover_table
from .evaluate import evaluate_review, read_result_rows, score_categories
from .gemini import MODEL_ID
from .prepare import index_by_id, read_csv, write_csv
from .schema import INSTRUCTION_HASH, INSTRUCTION_VERSION, json_loads, sha256, validate_label


def verify_snapshot(inventory, root, *, allow_partial=False):
    source_paths = [root / name for name in ("labels_by_text.jsonl", "comment_labels.jsonl", "manual_review_by_text.jsonl")]
    text_path, record_path, queue_path = source_paths
    entries = []
    for number, line in enumerate(text_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        entry = json_loads(line)
        if not isinstance(entry, dict) or not isinstance(entry.get("text_id"), str):
            raise ValueError(f"Invalid text label on line {number}")
        entries.append(entry)
    labels = index_by_id(entries, "text_id")
    expected = {key for key, row in inventory.texts.items() if row["content_status"] == "substantive"}
    if labels.keys() - expected or (not allow_partial and expected - labels.keys()):
        raise ValueError(f"Text coverage mismatch: {len(expected - labels.keys())} missing, {len(labels.keys() - expected)} extra")
    for key, entry in labels.items():
        if (
            entry.get("model") != MODEL_ID or entry.get("instruction_sha256") != INSTRUCTION_HASH
            or entry.get("instruction_version") != INSTRUCTION_VERSION or not entry.get("model_version")
        ):
            raise ValueError(f"Unexpected model or prompt provenance for {key}")
        validate_label(entry.get("label"), inventory.texts[key]["text"])
    queued = {}
    if queue_path.is_file():
        pending = [json_loads(line) for line in queue_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        queued = index_by_id(pending, "text_id")
        if queued and not allow_partial:
            raise ValueError("Some texts still require manual resolution")
        if queued.keys() - (expected - labels.keys()) or any(
            row.get("model") != MODEL_ID or row.get("instruction_sha256") != INSTRUCTION_HASH for row in queued.values()
        ):
            raise ValueError("Manual review queue does not match unresolved texts and their provenance")

    by_id = index_by_id(list(read_result_rows(record_path)), "comment_id")
    if by_id.keys() != {row["comment_id"] for row in inventory.links}:
        raise ValueError("Comment IDs in model export differ from the source inventory")
    for link in inventory.links:
        record = by_id[link["comment_id"]]
        annotation = labels.get(link["text_id"])
        status = link["content_status"]
        review = queued.get(link["text_id"])
        if status == "substantive":
            status = "classified" if annotation is not None else "needs_manual_review" if review is not None else "pending"
        if (
            any(field not in record or record[field] != value for field, value in link.items())
            or record.get("label_status") != status or "annotation" not in record or record["annotation"] != annotation
            or record.get("manual_review") != review
        ):
            raise ValueError(f"Inconsistent linkage or label for {link['comment_id']}")
    records = [by_id[link["comment_id"]] for link in inventory.links]
    return records, labels, {path.name: sha256(path.read_bytes()) for path in source_paths if path.is_file()}


def disagreement_table(evaluation, inventory):
    for item in evaluation["disagreements"]:
        key = item["text_id"]
        for field, values in item["fields"].items():
            yield {
                "text_id": key, "text": inventory.texts[key]["text"], "field": field,
                "comparison_type": "citation" if field.endswith("evidence") else "category",
                "human": render_value(values["gold"]), "gemini": render_value(values["model"]),
            }


def render_value(value):
    return " | ".join(value) if isinstance(value, list) else value


def publish_analysis(inventory, reference, review_path, processed, root, report):
    from .analysis_report import render

    records, labels, fingerprints = verify_snapshot(inventory, root)
    evaluation = evaluate_review(inventory, reference, records)
    if evaluation["substantive_agreement"] is None or evaluation["placeholder_exclusion"]["failed_text_ids"] or evaluation["placeholder_exclusion"]["unverified_text_ids"]:
        raise ValueError("Review text coverage or placeholder exclusion is incomplete")
    pairs = [(key, row["label"], labels[key]["label"]) for key, row in reference.items() if row["status"] == "substantive"]
    matrices, class_metrics, by_field = score_categories(pairs)
    differences = list(disagreement_table(evaluation["substantive_agreement"], inventory))

    settings = {
        "comments.csv": ("comment_id", "submitted_at"),
        "employee_wave.csv": ("employee_id", "wave", "team_id", "division", "wave_date", "responded"),
        "survey_at_wave.csv": ("employee_id", "wave", *TOPIC_TO_SCORE.values()),
        "exit_followup_90d.csv": ("employee_id", "wave", "as_of", "exit_within_90d"),
        "annual_turnover.csv": ("level", "group", "period_start", "period_end", "all_exits", "mean_headcount", "turnover"),
    }
    tables = {}
    for name, fields in settings.items():
        tables[name], fingerprints[name] = read_csv(processed / name, fields)
    linked, scored, people = scored_comments(
        records, tables["comments.csv"], tables["employee_wave.csv"], tables["survey_at_wave.csv"],
    )
    volumes, distributions = descriptive_tables(records, labels)
    topic_scores = topic_score_table(linked, scored, labels)
    turnover = turnover_table(linked, people, tables["exit_followup_90d.csv"], tables["annual_turnover.csv"], labels)
    different_fields = Counter(row["field"] for row in differences if row["comparison_type"] == "category")

    output = root / "analysis"
    manifest = {
        "input_sha256": inventory.checksums | fingerprints | {review_path.name: sha256(review_path.read_bytes())},
        "pipeline": {
            "unique_labeled_texts": len(labels), "comments": len(records),
            "row_status": dict(sorted(Counter(row["label_status"] for row in records).items())),
            "unconfirmed_author_comments": sum(row["author_status"] != "confirmed" for row in records),
            "usable_org_comments": sum(row["usable_team_text"] for row in records),
            "scored_linked_comments": len(linked), "scored_linked_employee_waves": len(scored),
            "model_review_flags": sum(annotation["label"]["needs_review"] for annotation in labels.values()),
            "unique_severity": dict(sorted(Counter(annotation["label"]["severity"] for annotation in labels.values()).items())),
            "model_versions": sorted({row["model_version"] for row in labels.values()}),
            "instruction_version": INSTRUCTION_VERSION,
        },
        "review": evaluation, "categorical_agreement": by_field, "categorical_disagreements": dict(sorted(different_fields.items())),
        "tables": {},
    }
    exports = {
        "review_confusion.csv": (("field", "label", "human", "gemini", "count"), matrices),
        "review_class_metrics.csv": (
            ("field", "category", "human_n", "gemini_n", "tp", "fp", "fn", "tn", "precision", "recall", "f1"), class_metrics,
        ),
        "review_disagreements.csv": (
            ("text_id", "text", "field", "comparison_type", "human", "gemini"), differences,
        ),
        "comment_volume.csv": (
            ("level", "wave", "division", "comments", "substantive_comments", "confirmed_authors",
             "substantive_confirmed_authors", "status"), volumes,
        ),
        "comment_distribution.csv": (
            ("level", "wave", "division", "dimension", "category", "comments", "unique_confirmed_authors",
             "substantive_denominator", "share", "status"), distributions,
        ),
        "topic_survey_links.csv": (
            ("wave", "topic", "question", "comparison", "mentions_employee_waves", "other_employee_waves",
             "mentions_unique_authors", "other_unique_authors", "mentions_mean", "other_mean", "difference", "status"), topic_scores,
        ),
        "comment_turnover_links.csv": (
            ("measure", "wave", "division", "period_start", "period_end", "exits", "turnover_denominator", "exit_rate",
             "substantive_comments", "unique_comment_authors", "negative_comments", "negative_share",
             "high_or_critical_comments", "high_or_critical_share", "comment_group_status"), turnover,
        ),
    }
    manifest["tables"] = {filename: len(rows) for filename, (_, rows) in exports.items()}
    markdown = render(manifest, class_metrics, volumes, distributions, topic_scores, turnover, output, report)
    output.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    for filename, (fields, rows) in exports.items():
        write_csv(output / filename, fields, rows, bom=True)
    write_text(output / "analysis_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_text(report, markdown)
    return manifest
