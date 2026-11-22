import csv
import io
import json
import os
import tempfile
import unittest
from collections import Counter
from datetime import date
from pathlib import Path
from shutil import copytree
from unittest.mock import Mock, patch

from src.pulse_comments.classify import run_classification
from src.pulse_comments.gemini import GeminiClient
from src.pulse_comments.prepare import Inventory, REVIEW_COLUMNS, read_csv, write_csv
from src.pulse_comments.schema import REVIEW_FIELDS
from src.pulse_comments.wave import MOCK, classify_wave, prepare_wave, report_wave


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "hr_pulse_comments_wave"
WAVE_DATE = date(2027, 2, 16)
REUSED = "Спасибо команде и руководителю."
FAILED = "После сбоя введённые данные не сохраняются."


def fixture_label(text):
    label = {
        "topics": ["другое"], "sentiment": "нейтральная", "severity": "отсутствует",
        "quit_intent": "свидетельств нет", "objects": ["не указан"],
        "actionability": "конкретного действия нет", "urgency": "не применимо", "ambiguity": ["отсутствует"],
        "confidence": .95, "evidence": text, "quit_evidence": "", "needs_review": False,
    }
    if text == REUSED:
        label.update(sentiment="положительная", objects=["команда", "руководитель"])
    elif text.startswith("Новый интерфейс"):
        label.update(topics=["инструменты", "нагрузка"], sentiment="отрицательная",
                     severity="средняя", ambiguity=["несколько тем"], objects=["информационная система"])
    elif text.startswith("Обожаю"):
        label.update(sentiment="неопределенная", ambiguity=["сарказм"], needs_review=True)
    elif text.startswith("Три недели"):
        label.update(topics=["инструменты"], severity="высокая", urgency="повышенная", needs_review=True)
    elif text.startswith("Я решил"):
        label.update(quit_intent="прямое упоминание", quit_evidence=text, needs_review=True)
    return label


class SimulatedTransport:
    def __init__(self):
        self.sent = Counter()
        self.format_repairs = 0

    def open(self, request, timeout):
        data = json.loads(request.data)
        text = json.loads(data["contents"][0]["parts"][0]["text"])["comment"]
        self.sent[text] += 1
        if "Previous response did not pass local validation" in data["systemInstruction"]["parts"][0]["text"]:
            self.format_repairs += 1
        label = {"topics": []} if text == FAILED else fixture_label(text)
        response = {"modelVersion": "simulated-2027-02", "responseId": f"mock-{sum(self.sent.values())}",
                    "candidates": [{"finishReason": "STOP", "content": {"parts": [
                        {"text": json.dumps(label, ensure_ascii=False)},
                    ]}}]}
        return io.BytesIO(json.dumps(response, ensure_ascii=False).encode("utf-8"))

    def classifier(self, rpm):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fixture-key"}):
            client = GeminiClient(rpm, opener=self, sleep=lambda _: None)
        return Mock(classify=lambda text: client.classify(text) | {"provider": MOCK})


class CommentWaveTests(unittest.TestCase):
    def test_new_wave_from_audit_to_aggregate_with_cache_resume_and_periodic_review(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            output, cache, seed = root / "wave", root / "shared.sqlite3", root / "previous"
            original = {path.name: path.read_bytes() for path in FIXTURE.iterdir()}
            inventory, state = prepare_wave(FIXTURE, output, WAVE_DATE, provider=MOCK)
            self.assertEqual((len(inventory.links), len(inventory.texts)), (18, 15))
            audit = [json.loads(line) for line in (output / "audit_issues.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertFalse(any(row["severity"] == "critical" for row in audit))
            self.assertTrue({"comment_placeholder", "orphan_comment_employee", "comment_pair_multiple"}
                            <= {row["rule"] for row in audit})

            reuse_key = next(key for key, row in inventory.texts.items() if row["text"] == REUSED)
            previous = Inventory(
                {reuse_key: inventory.texts[reuse_key]}, [row for row in inventory.links if row["text_id"] == reuse_key], {},
            )
            initial = SimulatedTransport()
            run_classification(previous, seed, 60, 1, client_factory=initial.classifier, provider=MOCK)
            archive = seed / "gemini_cache.sqlite3"
            seed_bytes = archive.read_bytes()
            self.assertEqual(initial.sent, Counter({REUSED: 1}))

            transport = SimulatedTransport()
            summary = classify_wave(inventory, state, output, cache, 60, seed=archive, client_factory=transport.classifier)
            self.assertNotIn(REUSED, transport.sent)
            self.assertNotIn("Нет комментариев.", transport.sent)
            self.assertNotIn("", transport.sent)
            self.assertEqual(sum(transport.sent.values()), 13)
            self.assertEqual(transport.sent[FAILED], 3)
            self.assertEqual(transport.format_repairs, 2)
            self.assertEqual(summary["coverage"]["classified_texts"], 11)
            self.assertEqual(summary["coverage"]["technical_review_texts"], 1)
            self.assertEqual(summary["coverage"]["rows_by_status"], {
                "classified": 14, "empty": 1, "placeholder": 2, "needs_manual_review": 1,
            })
            self.assertIsNone(summary["periodic_review"])
            self.assertEqual(summary["reported_model_versions"], ["simulated-2027-02"])
            volumes, _ = read_csv(output / "comment_volume.csv", ("classified_substantive_comments",))
            overall = next(row for row in volumes if row["level"] == "all")
            self.assertEqual((overall["substantive_comments"], overall["classified_substantive_comments"]), ("15", "14"))
            queue, _ = read_csv(output / "manual_followup.csv", ("text_id", "review_reason"))
            self.assertTrue(any("сарказм" in row["model_label_json"] for row in queue))
            self.assertTrue(any("severity_screen" in row["review_reason"] for row in queue))
            self.assertIn("Синтетическая имитация API", (output / "wave-summary.md").read_text(encoding="utf-8"))
            saved = (output / "comment_labels.jsonl").read_bytes()
            no_requests = Mock()
            resumed = classify_wave(inventory, state, output, cache, 60, seed=archive, client_factory=no_requests)
            no_requests.assert_not_called()
            self.assertEqual(archive.read_bytes(), seed_bytes)
            self.assertEqual(resumed, summary)
            self.assertEqual((output / "comment_labels.jsonl").read_bytes(), saved)

            sample = root / "short-control.json"
            sample.write_text(json.dumps([reuse_key]), encoding="utf-8")
            manual = root / "review.csv"
            gold = fixture_label(REUSED)
            values = {f"gold_{key}": "|".join(gold[key]) if isinstance(gold[key], list) else gold[key]
                      for key in REVIEW_FIELDS}
            write_csv(manual, REVIEW_COLUMNS, [{"text_id": reuse_key, "text": REUSED, "gold_status": "substantive", **values}])
            scored = report_wave(inventory, state, output, samples=sample, review=manual)["periodic_review"]
            self.assertEqual(scored["substantive_agreement"]["topics_micro_f1"], 1)
            self.assertEqual(original, {path.name: path.read_bytes() for path in FIXTURE.iterdir()})

    def test_critical_audit_stops_at_new_wave_cutoff(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            source = root / "raw"
            copytree(FIXTURE, source)
            path = source / "comments.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream, delimiter=";")
                fields, records = reader.fieldnames, list(reader)
            records[0]["submitted_at"] = "2027-02-17"
            write_csv(path, fields, records)
            with self.assertRaisesRegex(ValueError, "critical errors"):
                prepare_wave(source, root / "wave", WAVE_DATE, provider=MOCK)
            failures = (root / "wave" / "audit_issues.jsonl").read_text(encoding="utf-8")
            self.assertIn("after_wave_cutoff", failures)
            self.assertFalse((root / "wave" / "labels_by_text.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
