import json
from datetime import date

from src.pulse_comments.analysis_pipeline import verify_snapshot
from src.pulse_comments.classify import json_lines
from src.pulse_comments.prepare import build_inventory, index_by_id, read_csv, write_csv
from src.pulse_comments.schema import REVIEW_FIELDS, json_loads, sha256, validate_label
from src.pulse_quality.report import write_text


SOURCE_FIELDS = (
    "text_id", "text", "review_reason", "control_human_severity", "model_severity", "model_label_json",
    "model_version", "instruction_sha256", "model_processed_at_utc", "model_response_id",
)
HUMAN_FIELDS = ("decision", "corrections_json", "reviewer", "reviewed_on", "review_note")
FIELDS = (*SOURCE_FIELDS, *HUMAN_FIELDS)
TEMPLATE = "manual_followup.csv"


def followup_items(inventory, labels, differences, *, conservative=False):
    prior = {}
    for row in differences:
        if row["field"] != "severity":
            continue
        key = row["text_id"]
        if key in prior or key not in labels or row["gemini"] != labels[key]["label"]["severity"]:
            raise ValueError("Severity discrepancies differ from the saved annotations")
        prior[key] = row["human"]

    items = []
    for key, annotation in sorted(labels.items()):
        label = annotation["label"]
        reasons = []
        if label["needs_review"]:
            reasons.append("gemini_needs_review")
        if conservative and label["severity"] in {"средняя", "высокая", "критическая"}:
            reasons.append("severity_screen")
        if key in prior:
            reasons.append("severity_disagreement")
        if reasons:
            items.append({
                "text_id": key, "text": inventory.texts[key]["text"], "review_reason": "|".join(reasons),
                "control_human_severity": prior.get(key, ""), "model_severity": label["severity"],
                "model_label_json": json.dumps(label, ensure_ascii=False, sort_keys=True),
                "model_version": annotation["model_version"], "instruction_sha256": annotation["instruction_sha256"],
                "model_processed_at_utc": annotation["processed_at_utc"],
                "model_response_id": annotation.get("response_id") or "",
            })
    return items


def load_review_sources(raw, processed):
    directory = processed / "comment_analysis"
    inventory = build_inventory(processed / "comments.csv", raw / "comments.csv", raw / "employees.csv")
    _, labels, fingerprints = verify_snapshot(inventory, directory)
    differences, _ = read_csv(directory / "analysis" / "review_disagreements.csv", ("text_id", "field", "human", "gemini"))
    return directory, labels, followup_items(inventory, labels, differences), fingerprints["labels_by_text.jsonl"]


def prepare_followup(raw, processed):
    directory, _, items, _ = load_review_sources(raw, processed)
    path = directory / TEMPLATE
    write_followup_template(path, items)
    return path, len(items)


def write_followup_template(path, items):
    if path.exists():
        existing, _ = read_csv(path, FIELDS)
        if any(row[field].strip() for row in existing for field in HUMAN_FIELDS):
            raise ValueError("Keep the completed review copy before generating a new blank template")
    write_csv(path, FIELDS, items, bom=True)


def human_review(row, model):
    values = {field: row[field].strip() for field in HUMAN_FIELDS}
    decision = values["decision"]
    if not any(values.values()):
        return model.copy(), None
    if decision not in {"confirm", "correct"} or not values["reviewer"] or not values["reviewed_on"]:
        raise ValueError(f"{row['text_id']}: enter confirm/correct, reviewer and reviewed_on")
    try:
        reviewed_on = date.fromisoformat(values["reviewed_on"]).isoformat()
        if reviewed_on != values["reviewed_on"]:
            raise ValueError()
    except ValueError:
        raise ValueError(f"{row['text_id']}: reviewed_on must use YYYY-MM-DD") from None
    edits = json_loads(values["corrections_json"]) if values["corrections_json"] else {}
    if not isinstance(edits, dict) or edits.keys() - set(REVIEW_FIELDS):
        raise ValueError(f"{row['text_id']}: corrections_json must contain only classification fields")
    if decision == "confirm" and edits:
        raise ValueError(f"{row['text_id']}: use correct when supplying edits")
    changed = {key: {"before": model[key], "after": value} for key, value in edits.items() if model[key] != value}
    if decision == "correct" and (not changed or not values["review_note"]):
        raise ValueError(f"{row['text_id']}: a correction requires an actual change and a note")
    effective = model | edits
    validate_label({field: effective[field] for field in REVIEW_FIELDS}, row["text"], manual=True)
    return effective, {
        "decision": decision, "reviewer": values["reviewer"], "reviewed_on": reviewed_on,
        "note": values["review_note"], "changes": changed,
    }


def apply_followup(raw, processed, path):
    directory, labels, items, original_sha = load_review_sources(raw, processed)
    return apply_review_decisions(directory, labels, items, original_sha, path)


def apply_review_decisions(directory, labels, items, original_sha, path):
    filled, filled_sha = read_csv(path, FIELDS)
    decisions = index_by_id(filled, "text_id")
    expected = {row["text_id"]: row for row in items}
    if decisions.keys() != expected.keys():
        raise ValueError("The decisions file must reference the exact follow-up queue")
    reviewed = {}
    for key, row in decisions.items():
        if any(row[field] != expected[key][field] for field in SOURCE_FIELDS):
            raise ValueError(f"{key}: source fields differ from original")
        effective, provenance = human_review(row, labels[key]["label"])
        if provenance is not None:
            reviewed[key] = (effective, provenance | {
                "followup_sha256": filled_sha, "source_labels_sha256": original_sha,
                "source_response_id": row["model_response_id"], "review_reason": row["review_reason"],
            })

    output = directory / "reviewed_labels_overlay.jsonl"
    write_text(output, json_lines(
        {"text_id": key, "original": entry,
         "effective_label": reviewed[key][0] if key in reviewed else entry["label"],
         "human_review": reviewed[key][1] if key in reviewed else None}
        for key, entry in sorted(labels.items())
    ))
    return output, len(reviewed)
