import hashlib
import json
import math
from pathlib import Path


INSTRUCTION_VERSION = "1.0.0"
PROMPTS = Path(__file__).resolve().parents[2] / "prompts" / "comment_classification" / "v1"
INSTRUCTIONS = (PROMPTS / "instructions.md").read_text(encoding="utf-8")
SCHEMA = json.loads((PROMPTS / "schema.json").read_text(encoding="utf-8"))


def sha256(value):
    return hashlib.sha256(value).hexdigest()


INSTRUCTION_HASH = sha256((INSTRUCTIONS + json.dumps(SCHEMA, sort_keys=True, ensure_ascii=False)).encode("utf-8"))
REVIEW_FIELDS = tuple(name for name in SCHEMA["required"] if name not in {"confidence", "needs_review"})


class LabelValidationError(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


class DuplicateJSONField(ValueError):
    pass


def _verify(value, spec, field):
    kind = spec["type"]
    if kind == "array":
        if not isinstance(value, list):
            raise LabelValidationError("schema_type", f"{field}: expected an array")
        if not spec["minItems"] <= len(value) <= spec["maxItems"]:
            raise LabelValidationError("schema_range", f"{field}: expected {spec['minItems']}–{spec['maxItems']} items")
        for item in value:
            _verify(item, spec["items"], field)
        if spec["uniqueItems"] and len(set(value)) != len(value):
            raise LabelValidationError("invalid_category", f"{field}: repeated category")
    elif kind == "string":
        if not isinstance(value, str):
            raise LabelValidationError("schema_type", f"{field}: expected a string")
        if "enum" in spec and value not in spec["enum"]:
            raise LabelValidationError("invalid_category", f"{field}: use a category from the schema")
        if not spec.get("minLength", 0) <= len(value) <= spec.get("maxLength", math.inf):
            raise LabelValidationError("schema_range", f"{field}: invalid string length")
    elif kind == "number":
        if type(value) not in (int, float):
            raise LabelValidationError("schema_type", f"{field}: expected a number")
        if not spec["minimum"] <= value <= spec["maximum"]:
            raise LabelValidationError("schema_range", f"{field}: expected a number from {spec['minimum']} to {spec['maximum']}")
    elif kind == "boolean":
        if type(value) is not bool:
            raise LabelValidationError("schema_type", f"{field}: expected a boolean")
    else:
        raise ValueError(f"{field}: unsupported schema type")


def validate_label(label, text, *, manual=False):
    expected = REVIEW_FIELDS if manual else SCHEMA["required"]
    if not isinstance(label, dict):
        raise LabelValidationError("schema_type", "Expected a single JSON object")
    if label.keys() != set(expected):
        missing = ", ".join(field for field in expected if field not in label) or "none"
        extra_count = len(label.keys() - set(expected))
        raise LabelValidationError("schema_fields", f"Missing fields: {missing}. Extra fields: {extra_count}.")
    for name in expected:
        _verify(label[name], SCHEMA["properties"][name], name)
    for key, exclusive in (("topics", "другое"), ("objects", "не указан"), ("ambiguity", "отсутствует")):
        if exclusive in label[key] and len(label[key]) > 1:
            raise LabelValidationError("inconsistent_labels", f"{key}: exclusive category used with another category")
    if (len(label["topics"]) > 1) != ("несколько тем" in label["ambiguity"]):
        raise LabelValidationError("inconsistent_labels", "ambiguity: include the multi-topic flag exactly when topics has more than one item")
    if not label["evidence"].strip() or label["evidence"] not in text:
        raise LabelValidationError("citation_mismatch", "evidence: copy a nonempty verbatim fragment of the original comment")
    quit_signal = label["quit_intent"] != "свидетельств нет"
    if quit_signal:
        if not label["quit_evidence"].strip() or label["quit_evidence"] not in text:
            raise LabelValidationError("citation_mismatch", "quit_evidence: a quit signal requires a nonempty verbatim fragment")
    elif label["quit_evidence"]:
        raise LabelValidationError("inconsistent_labels", "quit_evidence: leave empty when there is no signal")
    if label["severity"] == "критическая" and label["urgency"] != "срочная":
        raise LabelValidationError("inconsistent_labels", "urgency: critical severity requires urgent review")
    if not manual:
        complex_case = {"сарказм", "недостаток контекста", "противоречивая оценка"}.intersection(label["ambiguity"])
        must_review = (
            label["confidence"] < 0.70 or complex_case or quit_signal
            or label["sentiment"] == "неопределенная" or label["severity"] in {"высокая", "критическая"}
        )
        if must_review and not label["needs_review"]:
            raise LabelValidationError("inconsistent_labels", "needs_review: expected true for an uncertain or consequential label")
    return label


def json_loads(value):
    def unique_keys(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise DuplicateJSONField("JSON object contains a repeated field")
        return obj

    return json.loads(value, object_pairs_hook=unique_keys)


def api_schema():
    # Shorter subset sent to the API. The full local validator remains authoritative.
    local_only = {"$schema", "uniqueItems", "minLength", "maxLength"}

    def simplify(spec):
        return {key: simplify(value) if isinstance(value, dict) else value
                for key, value in spec.items() if key not in local_only}

    return simplify(SCHEMA)
