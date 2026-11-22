import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from src.pulse_comments.analysis_pipeline import verify_snapshot
from src.pulse_comments.analytics import descriptive_tables, topic_score_table, turnover_table
from src.pulse_comments.classify import export_labels, run_classification
from src.pulse_comments.evaluate import agreement, evaluate_review, score_categories
from src.pulse_comments.gemini import (
    ENDPOINT, MAX_FORMAT_ATTEMPTS, MODEL_ID, GeminiClient, GeminiError, InvalidGeminiResponse, make_request,
    parse_response,
)
from src.pulse_comments.prepare import (
    REVIEW_COLUMNS, REVIEW_TEMPLATE, Inventory, build_inventory, export_inventory, read_csv, select_review_ids,
    validate_review, write_csv,
)
from src.pulse_comments.schema import INSTRUCTION_HASH, PROMPTS, REVIEW_FIELDS, json_loads, validate_label


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "Расписание удобное."


def example_label():
    return {
        "topics": ["условия"], "sentiment": "положительная", "severity": "отсутствует",
        "quit_intent": "свидетельств нет", "objects": ["рабочая среда"],
        "actionability": "конкретного действия нет", "urgency": "не применимо",
        "ambiguity": ["отсутствует"], "confidence": 0.9, "evidence": "удобное",
        "quit_evidence": "", "needs_review": False,
    }


def api_response(label):
    return {
        "modelVersion": "offline-test-revision", "responseId": "offline-test",
        "candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": json.dumps(label, ensure_ascii=False)},
        ]}}],
    }


def api_body(response):
    return io.BytesIO(json.dumps(response, ensure_ascii=False).encode("utf-8"))


class SchemaTests(unittest.TestCase):
    def test_known_label_and_manual_variant(self):
        label = example_label()
        self.assertEqual(validate_label(label, EXAMPLE), label)
        human_fields = {key: label[key] for key in REVIEW_FIELDS}
        self.assertEqual(validate_label(human_fields, EXAMPLE, manual=True), human_fields)

    def test_schema_rejects_bad_types_categories_and_fabricated_quote(self):
        for change in ({"confidence": True}, {"confidence": float("nan")}, {"topics": ["зарплата"]},
                       {"topics": ["условия", "условия"]}, {"evidence": "несуществующая фраза"},
                       {"quit_intent": "косвенное предположение"}, {"ambiguity": ["сарказм"]},
                       {"topics": ["условия", "нагрузка"]}, {"unexpected": True}):
            with self.subTest(fields=list(change)), self.assertRaises(ValueError):
                validate_label(example_label() | change, EXAMPLE)
        with self.assertRaises(ValueError):
            json_loads('{"score": 1, "score": 0}')

    def test_api_request_contains_only_comment_and_structured_output(self):
        request = make_request("offline-key", EXAMPLE)
        self.assertEqual(request.full_url, ENDPOINT)
        self.assertNotIn("offline-key", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "offline-key")
        payload = json.loads(request.data)
        self.assertEqual(json.loads(payload["contents"][0]["parts"][0]["text"]), {"comment": EXAMPLE})
        self.assertEqual(payload["generationConfig"]["responseFormat"]["text"]["mimeType"], "APPLICATION_JSON")
        self.assertNotIn("tools", payload)

    def test_bad_candidate_is_rejected(self):
        incomplete = api_response(example_label())
        incomplete["candidates"][0]["finishReason"] = "MAX_TOKENS"
        with self.assertRaisesRegex(InvalidGeminiResponse, "output_truncated: MAX_TOKENS"):
            parse_response(incomplete, EXAMPLE)

    def test_diagnostics_distinguish_json_schema_and_citation_errors(self):
        label = example_label()
        missing = {key: value for key, value in label.items() if key != "sentiment"} | {"offline-key": True}
        malformed_json = api_response(label)
        malformed_json["candidates"][0]["content"]["parts"] = [{"text": '{"x": "offline-key"'}]
        duplicate = api_response(label)
        duplicate["candidates"][0]["content"]["parts"] = [{"text": '{"x": 1, "x": 2}'}]
        for response, expected in (
            (malformed_json, "label_json",),
            (duplicate, "label_json: JSON contains a repeated field"),
            (api_response(missing), "schema_fields: Missing fields: sentiment. Extra fields: 1."),
            (api_response(label | {"sentiment": "offline-key"}), "invalid_category: sentiment"),
            (api_response(label | {"topics": "offline-key"}), "schema_type: topics"),
            (api_response(label | {"evidence": "offline-key"}), "citation_mismatch: evidence"),
            (api_response(label | {"confidence": 0.1}), "inconsistent_labels: needs_review"),
            (api_response(label) | {"modelVersion": ""}, "missing_model_version"),
        ):
            with self.subTest(reason=expected), self.assertRaises(InvalidGeminiResponse) as raised:
                parse_response(response, EXAMPLE)
            self.assertIn(expected, str(raised.exception))
            self.assertNotIn("offline-key", str(raised.exception))

    def test_format_retry_gives_precise_feedback_and_preserves_original_text(self):
        opener = Mock()
        opener.open.side_effect = [
            api_body(api_response(example_label() | {"evidence": "untrusted-text"})),
            api_body(api_response(example_label())),
        ]
        with patch.dict("os.environ", {"GEMINI_API_KEY": "offline-key"}):
            label = GeminiClient(60, opener=opener, sleep=Mock()).classify(EXAMPLE)
        self.assertEqual(opener.open.call_count, 2)
        self.assertEqual(label["label"], example_label())
        self.assertEqual(label["format_diagnostics"][0]["code"], "citation_mismatch")
        first, second = (json.loads(call.args[0].data) for call in opener.open.call_args_list)
        self.assertEqual(first["contents"], second["contents"])
        self.assertEqual(first["generationConfig"], second["generationConfig"])
        correction = second["systemInstruction"]["parts"][0]["text"]
        self.assertIn("citation_mismatch: evidence", correction)
        self.assertNotIn("untrusted-text", correction)
        self.assertNotIn("offline-key", correction)

    def test_format_retries_are_bounded_and_safety_block_is_not_retried(self):
        invalid = api_response(example_label() | {"sentiment": "unknown"})
        opener = Mock()
        opener.open.side_effect = [api_body(invalid) for _ in range(MAX_FORMAT_ATTEMPTS)]
        with patch.dict("os.environ", {"GEMINI_API_KEY": "offline-key"}):
            with self.assertRaises(InvalidGeminiResponse) as raised:
                GeminiClient(60, opener=opener, sleep=Mock()).classify(EXAMPLE)
            self.assertEqual(opener.open.call_count, MAX_FORMAT_ATTEMPTS)
            self.assertEqual(
                [(row["attempt"], row["code"]) for row in raised.exception.diagnostics],
                [(number, "invalid_category") for number in range(1, MAX_FORMAT_ATTEMPTS + 1)],
            )
            for block in (
                {"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "offline-key"}},
                {"candidates": [{"finishReason": "SAFETY"}]},
            ):
                opener = Mock()
                opener.open.return_value = api_body(block)
                with self.assertRaisesRegex(InvalidGeminiResponse, "provider_block.*SAFETY") as raised:
                    GeminiClient(60, opener=opener, sleep=Mock()).classify(EXAMPLE)
                self.assertEqual(opener.open.call_count, 1)
                self.assertNotIn("offline-key", str(raised.exception))

    def test_transient_errors_retry_and_long_quota_halts(self):
        attempts = Mock()
        attempts.open.side_effect = [
            HTTPError(ENDPOINT, 503, "temporary", Message(), io.BytesIO()),
            io.BytesIO(json.dumps(api_response(example_label())).encode("utf-8")),
        ]
        waits = Mock()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "offline-key"}):
            client = GeminiClient(60, opener=attempts, sleep=waits)
            result = client.classify(EXAMPLE)
            self.assertEqual(result["model"], MODEL_ID)
            self.assertEqual(result["model_version"], "offline-test-revision")
            self.assertEqual(attempts.open.call_count, 2)
            waits.assert_called()

            rate_limited = Mock()
            headers = Message()
            headers["Retry-After"] = "120"
            rate_limited.open.side_effect = HTTPError(ENDPOINT, 429, "quota", headers, io.BytesIO())
            with self.assertRaisesRegex(GeminiError, "quota"):
                GeminiClient(60, opener=rate_limited, sleep=Mock()).classify(EXAMPLE)
            self.assertEqual(rate_limited.open.call_count, 1)


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = build_inventory(
            ROOT / "data/processed/comments.csv", ROOT / "data/raw/comments.csv", ROOT / "data/raw/employees.csv",
        )
        cls.sample = select_review_ids(cls.inventory, PROMPTS / "review_sample_ids.json")

    def test_distinct_texts_and_lineage_cover_every_comment(self):
        inventory = self.inventory
        self.assertEqual(sum(row["occurrences"] for row in inventory.texts.values()), len(inventory.links))
        keys = {row["text_id"] for row in inventory.links}
        self.assertEqual(keys, set(inventory.texts))
        self.assertEqual(len({inventory.texts[key]["text"] for key in keys}), len(keys))
        self.assertEqual(len({link["comment_id"] for link in inventory.links}), len(inventory.links))
        flagged = [link for link in inventory.links
                   if link["author_status"] != "confirmed" or link["content_status"] != "substantive"]
        self.assertTrue(flagged)
        self.assertFalse(any(link["usable_team_text"] for link in flagged))

    def test_manual_template_is_blind_and_cannot_unlock_generation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            export_inventory(self.inventory, output, self.sample)
            path = output / REVIEW_TEMPLATE
            rows, _ = read_csv(path, REVIEW_COLUMNS)
            self.assertEqual(len(rows), 30)
            self.assertEqual(len({row["text"] for row in rows}), 30)
            self.assertTrue(all(set(row) == set(REVIEW_COLUMNS) for row in rows))
            self.assertTrue(all(not row[key] for row in rows for key in REVIEW_COLUMNS[2:]))
            with self.assertRaisesRegex(ValueError, "30 review rows"):
                validate_review(path, self.inventory, self.sample)

    def test_frozen_review_sample_has_one_placeholder_and_29_substantive_texts(self):
        statuses = [self.inventory.texts[key]["content_status"] for key in self.sample]
        self.assertEqual(statuses.count("placeholder"), 1)
        self.assertEqual(statuses.count("substantive"), 29)

    def test_cache_survives_partial_generation_and_skips_saved_items(self):
        inventory = Inventory(
            texts={
                "a": {"text": EXAMPLE, "content_status": "substantive"},
                "b": {"text": EXAMPLE + " Спасибо.", "content_status": "substantive"},
                "p": {"text": "Нет комментариев.", "content_status": "placeholder"},
            },
            links=[
                {"comment_id": "1", "text_id": "a", "content_status": "substantive"},
                {"comment_id": "2", "text_id": "a", "content_status": "substantive"},
                {"comment_id": "3", "text_id": "b", "content_status": "substantive"},
                {"comment_id": "4", "text_id": "p", "content_status": "placeholder"},
            ],
            checksums={},
        )
        annotation = parse_response(api_response(example_label()), EXAMPLE)
        first = Mock()
        first.classify.side_effect = [annotation, GeminiError("offline test interruption")]
        second = Mock()
        second.classify.return_value = annotation
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            with self.assertRaises(GeminiError):
                run_classification(inventory, output, 5, 3, client_factory=Mock(return_value=first))
            cached = (output / "labels_by_text.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(row)["text_id"] for row in cached], ["a"])
            self.assertEqual(run_classification(
                inventory, output, 5, 3, client_factory=Mock(return_value=second)
            ), (2, 0, 0))
            second.classify.assert_called_once_with(inventory.texts["b"]["text"])
            lines = (output / "comment_labels.jsonl").read_text(encoding="utf-8").splitlines()
            row_a, repeated, _, placeholder = [json.loads(line) for line in lines]
            self.assertEqual(repeated["annotation"], row_a["annotation"])
            self.assertEqual((placeholder["label_status"], placeholder["annotation"]), ("placeholder", None))
            unused = Mock()
            self.assertEqual(run_classification(inventory, output, 5, 3, client_factory=unused), (2, 0, 0))
            unused.assert_not_called()

    def test_unrepairable_text_is_queued_and_other_texts_continue_with_legacy_cache(self):
        texts = {key: {"text": f"{EXAMPLE} {key}", "content_status": "substantive"} for key in ("a", "b", "c")}
        links = [{"comment_id": str(index), "text_id": key, "content_status": "substantive"}
                 for index, key in enumerate(("a", "b", "b", "c"), 1)]
        inventory = Inventory(texts, links, {})
        annotation = parse_response(api_response(example_label()), EXAMPLE)
        cached = annotation | {"text_id": "a"}
        client = Mock()
        client.classify.side_effect = [InvalidGeminiResponse("citation_mismatch", "evidence: copy a fragment"), annotation]

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            db_path = output / "gemini_cache.sqlite3"
            with closing(sqlite3.connect(db_path)) as con, con:
                con.execute("""CREATE TABLE labels (
                    text_id TEXT NOT NULL, model TEXT NOT NULL, instruction_hash TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (text_id, model, instruction_hash)
                )""")
                con.execute("INSERT INTO labels VALUES (?, ?, ?, ?)",
                            ("a", MODEL_ID, INSTRUCTION_HASH, json.dumps(cached)))
            with patch("sys.stderr", new_callable=io.StringIO) as message:
                progress = run_classification(inventory, output, 5, 2, client_factory=Mock(return_value=client))
                self.assertIn("needs manual review b", message.getvalue().lower())
            self.assertEqual(progress, (2, 1, 0))
            self.assertEqual([call.args[0] for call in client.classify.call_args_list],
                             [texts["b"]["text"], texts["c"]["text"]])
            exported = [json.loads(line) for line in (output / "comment_labels.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(exported[0]["annotation"], cached)
            self.assertEqual([row["label_status"] for row in exported],
                             ["classified", "needs_manual_review", "needs_manual_review", "classified"])
            self.assertIsNone(exported[1]["annotation"])
            self.assertEqual(exported[1]["manual_review"], exported[2]["manual_review"])
            queue = json.loads((output / "manual_review_by_text.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(queue["diagnostics"][0]["code"], "citation_mismatch")
            unused = Mock()
            self.assertEqual(run_classification(inventory, output, 5, 2, client_factory=unused), (2, 1, 0))
            unused.assert_not_called()


class ReviewEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.inventory = Inventory(
            texts={"a": {"text": EXAMPLE}, "p": {"text": "Без комментариев."}},
            links=[
                {"comment_id": "1", "text_id": "a"},
                {"comment_id": "2", "text_id": "a"},
                {"comment_id": "3", "text_id": "p"},
                {"comment_id": "4", "text_id": "p"},
            ],
            checksums={},
        )
        self.reference = {
            "a": {"status": "substantive", "label": {key: example_label()[key] for key in REVIEW_FIELDS}},
            "p": {"status": "placeholder", "label": None},
        }
        annotation = {"text_id": "a", "label": example_label()}
        self.results = [
            link | ({"label_status": "classified", "annotation": annotation} if link["text_id"] == "a"
                    else {"label_status": "placeholder", "annotation": None})
            for link in self.inventory.links
        ]

    def test_placeholder_needs_only_status_and_substantive_needs_categories(self):
        human_row = {
            f"gold_{key}": "|".join(value) if isinstance(value, list) else value
            for key, value in self.reference["a"]["label"].items()
        }
        rows = [
            {"text_id": "a", "text": EXAMPLE, "gold_status": "substantive", **human_row},
            {"text_id": "p", "text": "Без комментариев.", "gold_status": "placeholder"},
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "test_reference.csv"
            write_csv(path, REVIEW_COLUMNS, rows)
            self.assertEqual(validate_review(path, self.inventory, ["a", "p"]), self.reference)
            write_csv(path, REVIEW_COLUMNS, [rows[0] | {"gold_sentiment": ""}, rows[1]])
            with self.assertRaisesRegex(ValueError, "row 1: sentiment"):
                validate_review(path, self.inventory, ["a", "p"])
            write_csv(path, REVIEW_COLUMNS, [rows[0], rows[1] | {"gold_sentiment": "нейтральная"}])
            with self.assertRaisesRegex(ValueError, "row 2: gold_"):
                validate_review(path, self.inventory, ["a", "p"])

    def test_placeholder_and_repeated_rows_do_not_inflate_agreement(self):
        result = evaluate_review(self.inventory, self.reference, self.results)
        self.assertEqual(result["placeholder_exclusion"]["correctly_excluded"], 1)
        self.assertEqual(result["substantive_agreement"]["texts"], 1)
        self.assertEqual(result["substantive_agreement"]["sentiment_accuracy"], 1)
        self.assertIsNone(result["substantive_agreement"]["serious_problem_recall"])

    def test_bad_exclusion_and_partial_coverage_are_visible_separately(self):
        result = evaluate_review(self.inventory, self.reference, self.results[:-1])
        self.assertEqual(result["placeholder_exclusion"]["unverified_text_ids"], ["p"])
        self.assertEqual(result["substantive_agreement"]["texts"], 1)
        self.results[1]["label_status"] = "pending"
        self.results[2]["annotation"] = {"label": {}}
        result = evaluate_review(self.inventory, self.reference, self.results)
        self.assertEqual(result["placeholder_exclusion"]["failed_text_ids"], ["p"])
        self.assertEqual(result["incomplete_substantive_text_ids"], ["a"])
        self.assertIsNone(result["substantive_agreement"])

    def test_multilabel_precision_recall_and_empty_single_class(self):
        gold = self.reference["a"]["label"]
        first = example_label() | {"topics": ["условия", "нагрузка"]}
        second = example_label() | {"topics": ["нагрузка"]}
        pairs = [("a", gold, first), ("b", gold, second)]
        metrics = agreement(pairs)
        self.assertAlmostEqual(metrics["topics_micro_precision"], 1 / 3)
        self.assertAlmostEqual(metrics["topics_micro_recall"], 1 / 2)
        self.assertAlmostEqual(metrics["topics_micro_f1"], 2 / 5)
        matrices, classes, summary = score_categories(pairs)
        positive = next(row for row in classes if row["category"] == "прямое упоминание")
        self.assertIsNone(positive["recall"])
        self.assertIsNone(positive["precision"])
        self.assertIsNone(summary["quit_intent"]["kappa"])
        cell = next(row for row in matrices if row["label"] == "нагрузка" and row["human"] == 0 and row["gemini"] == 1)
        self.assertEqual(cell["count"], 2)


def comment_record(identifier, author, text_id, *, wave="2026-03", usable=True, confirmed=True):
    return {
        "comment_id": identifier, "employee_id": author, "text_id": text_id, "wave": wave, "division": "IT",
        "content_status": "substantive", "usable_team_text": usable,
        "author_status": "confirmed" if confirmed else "unknown_author",
    }


class CommentAnalysisTests(unittest.TestCase):
    def test_complete_snapshot_detects_inconsistent_expansion(self):
        labels = {"t": parse_response(api_response(example_label()), EXAMPLE) | {"text_id": "t"}}
        inventory = Inventory(
            {"t": {"text": EXAMPLE, "content_status": "substantive"}, "p": {"content_status": "placeholder"}},
            [comment_record("1", "a", "t"), comment_record("2", "a", "p", usable=False) | {"content_status": "placeholder"}],
            {},
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            export_labels(inventory, output, labels, {})
            original, verified, _ = verify_snapshot(inventory, output)
            self.assertEqual(labels, verified)
            self.assertEqual(original[1]["label_status"], "placeholder")
            original[0]["annotation"] = None
            (output / "comment_labels.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in original), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Inconsistent linkage"):
                verify_snapshot(inventory, output)

    def test_description_separates_authors_records_and_suppresses_exclusive_cells(self):
        rows = [
            comment_record(str(i), str(i), "positive" if i < 13 else "negative")
            for i in range(15)
        ]
        rows.extend([
            comment_record("16", "0", "positive"),
            comment_record("17", "unknown", "positive", usable=False, confirmed=False),
        ])
        labels = {
            "positive": {"label": example_label()},
            "negative": {"label": example_label() | {"sentiment": "отрицательная"}},
        }
        volumes, distribution = descriptive_tables(rows, labels)
        all_volume = next(row for row in volumes if row["level"] == "all")
        self.assertEqual((all_volume["substantive_comments"], all_volume["substantive_confirmed_authors"]), (17, 15))
        div_volume = next(row for row in volumes if row["level"] == "division")
        self.assertEqual(div_volume["substantive_comments"], 16)
        tones = {row["category"]: row for row in distribution if row["level"] == "all" and row["dimension"] == "sentiment"}
        self.assertEqual(tones["отрицательная"]["status"], "suppressed_cell")
        self.assertEqual(tones["положительная"]["status"], "suppressed_complement")
        self.assertIsNone(tones["отрицательная"]["comments"])

    def test_score_analysis_counts_one_observation_per_employee_wave(self):
        sample = [comment_record("a", "1", "pay"), comment_record("b", "1", "pay"),
                  comment_record("c", "2", "balance")]
        scores = {(r["employee_id"], r["wave"]): dict.fromkeys(
            ("q_conditions", "q_recognition", "q_growth", "q_manager", "q_pay", "q_balance", "q_goals", "q_tools"),
            "4" if r["employee_id"] == "1" else "8",
        ) for r in sample}
        labels = {"pay": {"label": {"topics": ["оплата"], "sentiment": "отрицательная"}},
                  "balance": {"label": {"topics": ["нагрузка"], "sentiment": "положительная"}}}
        with patch("src.pulse_comments.analytics.SMALL_GROUP", 1):
            output = topic_score_table(sample, scores, labels)
        pay = next(row for row in output if not row["wave"] and row["topic"] == "оплата"
                   and row["comparison"] == "problem_comment")
        self.assertEqual(pay["mentions_employee_waves"], 1)
        self.assertEqual(pay["other_employee_waves"], 1)
        self.assertEqual(pay["difference"], -4)

    def test_followup_view_uses_comments_as_of_response(self):
        sample = [comment_record("1", "e1", "t") | {"submitted_at": "2026-03-16"},
                  comment_record("2", "e1", "t") | {"submitted_at": "2026-03-17"}]
        workforce = {("e1", "2026-03"): {"division": "IT"}}
        outcomes = [{"employee_id": "e1", "wave": "2026-03", "as_of": "2026-03-16", "exit_within_90d": "False"}]
        labels = {"t": {"label": example_label()}}
        with patch("src.pulse_comments.analytics.SMALL_GROUP", 1):
            views = turnover_table(sample, workforce, outcomes, [], labels)
        self.assertEqual(views[0]["substantive_comments"], 1)
        self.assertEqual(views[0]["exit_rate"], 0)


if __name__ == "__main__":
    unittest.main()
