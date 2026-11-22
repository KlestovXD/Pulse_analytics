from collections import Counter, defaultdict

from .schema import REVIEW_FIELDS, SCHEMA, json_loads, validate_label


SCALAR_FIELDS = ("sentiment", "severity", "urgency", "actionability", "quit_intent")
MULTILABEL_FIELDS = ("topics", "objects", "ambiguity")
SEVERITY_LEVELS = SCHEMA["properties"]["severity"]["enum"]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def read_result_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json_loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("comment_id"), str):
                raise ValueError(f"Results row {number}: expected a comment record with an ID")
            yield row


def agreement(pairs):
    size = len(pairs)
    sentiment_matches = exact_topics = shared_topics = model_topics = gold_topics = 0
    serious_found = serious_gold = no_quit_signal = 0
    unsupported_quit = Counter()
    differences = []
    severity_changes = Counter()
    for key, gold, model in pairs:
        gold_set, model_set = set(gold["topics"]), set(model["topics"])
        exact_topics += gold_set == model_set
        shared_topics += len(gold_set & model_set)
        model_topics += len(model_set)
        gold_topics += len(gold_set)
        sentiment_matches += gold["sentiment"] == model["sentiment"]
        shift = SEVERITY_LEVELS.index(model["severity"]) - SEVERITY_LEVELS.index(gold["severity"])
        severity_changes["higher" if shift > 0 else "lower" if shift < 0 else "same"] += 1
        if gold["severity"] in {"высокая", "критическая"}:
            serious_gold += 1
            serious_found += model["severity"] in {"высокая", "критическая"}
        if gold["quit_intent"] == "свидетельств нет":
            no_quit_signal += 1
            if model["quit_intent"] != gold["quit_intent"]:
                unsupported_quit[model["quit_intent"]] += 1
        differing = {}
        for field in REVIEW_FIELDS:
            left, right = gold[field], model[field]
            equal = set(left) == set(right) if isinstance(left, list) else left == right
            if not equal:
                differing[field] = {"gold": left, "model": right}
        if differing:
            differences.append({"text_id": key, "fields": differing})
    return {
        "texts": size,
        "sentiment_accuracy": sentiment_matches / size,
        "topics_exact_match": exact_topics / size,
        "topics_micro_precision": ratio(shared_topics, model_topics),
        "topics_micro_recall": ratio(shared_topics, gold_topics),
        "topics_micro_f1": ratio(2 * shared_topics, model_topics + gold_topics),
        "topics_shared": shared_topics, "topics_extra": model_topics - shared_topics,
        "topics_missed": gold_topics - shared_topics,
        "severity_direction": {direction: severity_changes[direction] for direction in ("same", "lower", "higher")},
        "serious_problem_recall": serious_found / serious_gold if serious_gold else None,
        "serious_problems_in_reference": serious_gold,
        "unsupported_quit_signals": dict(unsupported_quit),
        "quit_false_positive_rate": unsupported_quit.total() / no_quit_signal if no_quit_signal else None,
        "no_quit_signal_in_reference": no_quit_signal,
        "disagreements": differences,
    }


def score_categories(pairs):
    matrices, class_metrics, summary = [], [], {}
    for field in SCALAR_FIELDS:
        classes = SCHEMA["properties"][field]["enum"]
        cells = Counter((gold[field], model[field]) for _, gold, model in pairs)
        true = Counter(gold[field] for _, gold, _ in pairs)
        predicted = Counter(model[field] for _, _, model in pairs)
        size = len(pairs)
        exact = sum(cells[category, category] for category in classes)
        # Linear weights preserve the order of severity levels. Other fields are nominal.
        observed, expected = 0, 0
        for i, gold in enumerate(classes):
            for j, model in enumerate(classes):
                count = cells[gold, model]
                matrices.append({"field": field, "label": "", "human": gold, "gemini": model, "count": count})
                penalty = abs(i - j) if field == "severity" else int(i != j)
                observed += penalty * count
                expected += penalty * true[gold] * predicted[model]
        summary[field] = {
            "matches": exact, "accuracy": ratio(exact, size),
            "kappa": 1 - observed * size / expected if expected else None,
        }
        for category in classes:
            tp = cells[category, category]
            class_metrics.append(score_class(field, category, tp, predicted[category] - tp, true[category] - tp, size))

    for field in MULTILABEL_FIELDS:
        classes = SCHEMA["properties"][field]["items"]["enum"]
        for category in classes:
            cells = Counter((category in gold[field], category in model[field]) for _, gold, model in pairs)
            for human in (False, True):
                for gemini in (False, True):
                    matrices.append({
                        "field": field, "label": category, "human": int(human), "gemini": int(gemini),
                        "count": cells[human, gemini],
                    })
            class_metrics.append(score_class(
                field, category, cells[True, True], cells[False, True], cells[True, False], len(pairs),
            ))
    return matrices, class_metrics, summary


def score_class(field, category, tp, fp, fn, size):
    return {
        "field": field, "category": category, "human_n": tp + fn, "gemini_n": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": size - tp - fp - fn,
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn) if tp + fn else None,
    }


def evaluate_review(inventory, reference, result_rows):
    expected = {
        row["comment_id"]: row["text_id"] for row in inventory.links if row["text_id"] in reference
    }
    expected_counts = Counter(expected.values())
    observed, seen = defaultdict(list), set()
    for row in result_rows:
        comment = row["comment_id"]
        if comment not in expected:
            if row.get("text_id") in reference:
                raise ValueError("Unexpected comment ID points to a review text")
            continue
        key = expected[comment]
        if comment in seen or row.get("text_id") != key or "annotation" not in row or "label_status" not in row:
            raise ValueError(f"Invalid or repeated results row for {comment}")
        seen.add(comment)
        observed[key].append((row["label_status"], row["annotation"]))

    excluded, incorrect, unverified, pending, pairs = [], [], [], [], []
    for key, gold in reference.items():
        records = observed[key]
        complete = len(records) == expected_counts[key] and bool(records)
        if gold["status"] == "placeholder":
            if any(status != "placeholder" or annotation is not None for status, annotation in records):
                incorrect.append(key)
            elif complete:
                excluded.append(key)
            else:
                unverified.append(key)
            continue
        if not complete or any(status != "classified" for status, _ in records):
            pending.append(key)
            continue
        annotation = records[0][1]
        if (not isinstance(annotation, dict) or annotation.get("text_id") != key
                or any(other != annotation for _, other in records)):
            raise ValueError(f"Missing or inconsistent model annotation for {key}")
        model = validate_label(annotation.get("label"), inventory.texts[key]["text"])
        pairs.append((key, gold["label"], model))

    return {
        "placeholder_exclusion": {
            "texts_in_reference": len(excluded) + len(incorrect) + len(unverified),
            "correctly_excluded": len(excluded),
            "failed_text_ids": incorrect,
            "unverified_text_ids": unverified,
        },
        "substantive_texts": len(pairs) + len(pending),
        "classified_substantive_texts": len(pairs),
        "incomplete_substantive_text_ids": pending,
        "substantive_agreement": agreement(pairs) if pairs and not pending else None,
    }
