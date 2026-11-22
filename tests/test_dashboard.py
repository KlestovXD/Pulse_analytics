import json
import re
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from src.pulse_comments.prepare import read_csv, write_csv
from src.pulse_dashboard.build import benjamini_hochberg, build, build_snapshot
from src.pulse_dashboard.review import FIELDS, HUMAN_FIELDS, apply_followup, human_review, load_review_sources


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot, cls.families = build_snapshot(PROCESSED)

    def test_headlines_dates_and_division_filters_read_prepared_values(self):
        data = self.snapshot
        source, _ = read_csv(PROCESSED / "wave_metrics.csv", ("group", "wave", "level"))
        for item in source:
            scope = "" if item["level"] == "total" else item["group"]
            result = data["survey"][scope][item["wave"]]
            for metric in ("participation", "engagement", "recommend_signal"):
                self.assertEqual(result[metric][0], float(item[metric]))
                self.assertEqual(result[metric][1:], [float(item[metric + "_lo"]), float(item[metric + "_hi"])])
        self.assertEqual([(w["id"], w["date"]) for w in data["waves"]], [
            ("2026-03", "2026-03-16"), ("2026-06", "2026-06-15"), ("2026-09", "2026-09-14"),
        ])
        end = data["survey"][""]["2026-09"]
        self.assertEqual(round(end["participation"][0], 1), 74.3)
        self.assertEqual(round(end["engagement"][0], 2), 62.62)
        self.assertEqual(round(end["recommend_signal"][0], 1), -9.9)
        self.assertEqual(round(data["annual"]["value"][""], 2), 14.42)
        self.assertEqual(round(data["annual"]["voluntary"][""], 2), 13.12)
        self.assertEqual((data["annual"]["start"], data["annual"]["end"]), ("2025-10-01", "2026-09-30"))

    def test_bh_is_computed_on_full_families(self):
        adjusted = benjamini_hochberg({"c": .20, "a": .01, "b": .04})
        self.assertAlmostEqual(adjusted["a"], .03)
        self.assertAlmostEqual(adjusted["b"], .06)
        self.assertAlmostEqual(adjusted["c"], .20)
        self.assertEqual(self.families, {"2026-06": 34, "2026-09": 33})
        status = {(r["wave"], r["team"]): (r["confirmed"], r["q"]) for r in self.snapshot["signals"]}
        self.assertEqual(status, {
            ("2026-06", "T16"): (True, .018),
            ("2026-09", "T06"): (False, .637),
            ("2026-09", "T34"): (False, .637),
        })

    def test_team_suppression_is_stable_and_has_a_complement(self):
        teams = {t["id"]: t for t in self.snapshot["teams"]}
        source, _ = read_csv(PROCESSED / "team_metrics.csv", ("team_id", "publication_status"))
        primary = {r["team_id"] for r in source if r["publication_status"] != "available"}
        hidden = {key for key, team in teams.items() if all(value is None for value in team["waves"].values())}
        self.assertLessEqual(primary, hidden)
        self.assertGreater(len(hidden), len(primary))
        members = defaultdict(set)
        for team in teams.values():
            members[team["division"]].add(team["id"])
            if team["id"] in hidden:
                self.assertEqual(set(team), {"id", "division", "waves"})
                self.assertTrue(all(value is None for value in team["waves"].values()))
            else:
                self.assertTrue(all(row["respondents"] >= 10 for row in team["waves"].values()))
        self.assertTrue(all(len(group & hidden) != 1 for group in members.values()))
        self.assertTrue(all(row["team"] not in hidden for row in self.snapshot["signals"]))

    def test_small_comment_cells_and_their_complements_are_absent(self):
        source, _ = read_csv(PROCESSED / "comment_analysis/analysis/comment_distribution.csv", ("status",))
        columns = {
            "sentiment": "tone",
            "topic_in_positive_comment": "positive_topics",
            "topic_in_negative_comment": "negative_topics",
        }
        for row in source:
            if row["level"] not in {"wave", "division_wave"} or row["dimension"] not in columns:
                continue
            cell = self.snapshot["comments"][row["wave"]][row["division"]][columns[row["dimension"]]]
            if row["category"] not in cell:
                continue
            actual = cell[row["category"]]
            if row["status"] != "available":
                self.assertIsNone(actual)
            elif actual is not None:
                self.assertEqual(actual, round(100 * float(row["share"]), 1))
        for scopes in self.snapshot["comments"].values():
            for field in columns.values():
                for key, total in scopes[""][field].items():
                    if total is not None:
                        self.assertNotEqual(sum(scopes[d][field][key] is None for d in self.snapshot["divisions"]), 1)
            for scope in scopes.values():
                self.assertNotEqual(sum(value is None for value in scope["tone"].values()), 1)
        september = self.snapshot["comments"]["2026-09"][""]
        self.assertEqual(september["positive_topics"]["развитие"], round(100 * 43 / 379, 1))
        self.assertEqual(september["negative_topics"]["нагрузка"], round(100 * 43 / 379, 1))
        self.assertIsNone(september["negative_topics"]["оплата"])  # The mixed remainder would expose six records.
        self.assertIsNone(september["positive_topics"]["руководитель"])  # The neutral remainder also has six records.

    def test_market_shows_only_matching_public_bank_disclosures(self):
        banks = {item["name"]: item for item in self.snapshot["market"]}
        self.assertEqual({name: row["rates"] for name, row in banks.items()}, {
            "Банк ДОМ.РФ": {"value": 21.1, "voluntary": 18.6},
            "Банк ВТБ": {"value": 27, "voluntary": 19},
        })
        self.assertTrue(all(bank["url"].startswith("https://") for bank in banks.values()))

    def test_generated_file_embeds_only_the_public_snapshot_and_no_external_assets(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory) / "index.html"
            build(PROCESSED, target)
            first = target.read_bytes()
            build(PROCESSED, target)
            self.assertEqual(first, target.read_bytes())
            html = first.decode("utf-8")
        data = re.search(r'<script id="snapshot" type="application/json">(.*?)</script>', html, re.S)
        self.assertEqual(json.loads(data[1]), self.snapshot)
        self.assertNotRegex(html, r"EMP[0-9]+|employee_id|comment_id|text_id")
        self.assertNotRegex(html, r'<(?:script|link|img)[^>]+\b(?:src|href)="https?:')


class FollowupTests(unittest.TestCase):
    def test_production_queue_uses_union_and_keeps_only_confirmed_prior_severity(self):
        _, _, items, _ = load_review_sources(RAW, PROCESSED)
        self.assertEqual(len(items), 12)
        self.assertEqual(len({r["text_id"] for r in items}), len(items))
        self.assertEqual(sum("gemini_needs_review" in r["review_reason"] for r in items), 10)
        self.assertEqual(sum("severity_disagreement" in r["review_reason"] for r in items), 3)
        self.assertEqual(sum(bool(r["control_human_severity"]) for r in items), 3)
        self.assertTrue(all(not (row.keys() & set(HUMAN_FIELDS)) for row in items))

    def test_apply_keeps_original_label_with_review_provenance(self):
        text = "Доступ отсутствует."
        original = {
            "topics": ["инструменты"], "sentiment": "отрицательная", "severity": "средняя",
            "quit_intent": "свидетельств нет", "objects": ["информационная система"],
            "actionability": "есть конкретная проблема для проверки", "urgency": "обычная",
            "ambiguity": ["отсутствует"], "evidence": "Доступ отсутствует", "quit_evidence": "",
            "confidence": .9, "needs_review": False,
        }
        row = dict.fromkeys(FIELDS, "") | {"text_id": "fixture", "text": text, "review_reason": "severity_disagreement"}
        decision = row | {
            "decision": "correct", "corrections_json": '{"severity":"высокая"}', "reviewer": "reviewer-fixture",
            "reviewed_on": "2026-11-22", "review_note": "Проверка последствий",
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            input_path = root / "decisions.csv"
            write_csv(input_path, FIELDS, [decision])
            model = {"fixture": {"label": original}}
            with patch("src.pulse_dashboard.review.load_review_sources", return_value=(root, model, [row], "original-fixture-sha")):
                artifact, size = apply_followup(RAW, PROCESSED, input_path)
            output = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertEqual(size, 1)
        self.assertEqual(output["original"]["label"]["severity"], "средняя")
        self.assertEqual(output["effective_label"]["severity"], "высокая")
        self.assertEqual(output["human_review"]["changes"]["severity"], {"before": "средняя", "after": "высокая"})
        self.assertEqual(output["human_review"]["source_labels_sha256"], "original-fixture-sha")
        self.assertEqual(original["severity"], "средняя")
        with self.assertRaisesRegex(ValueError, "classification fields"):
            human_review(decision | {"corrections_json": '{"confidence":1}'}, original)


if __name__ == "__main__":
    unittest.main()
