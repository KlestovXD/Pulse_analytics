import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.pulse_quality.audit import run_audit
from src.pulse_quality.contract import SCORES, TABLES
from src.pulse_quality.report import write_reports


ROOT = Path(__file__).resolve().parents[1]
FIRST, SECOND, THIRD = "EMP00001", "EMP00002", "EMP00003"
UNKNOWN = "EMP99999"


def response(identifier, employee, wave="2026-03", day="2026-03-16", **scores):
    values = {col: "7" for col in SCORES} | scores
    return [identifier, wave, employee, day, *(values[col] for col in SCORES)]


def comment(identifier, employee, day="16.03.2026", team="T01", text="Один; два"):
    return [identifier, "2026-03", employee, team, "Розничный блок", day, text]


class AuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.raw = self.directory / "raw"
        self.raw.mkdir()
        self.rows = {
            "employees": [
                [FIRST, "T01", "Розничный блок", "Руководитель группы", "офис", "2024-01-01", "EMP00000"],
                [SECOND, "T01", "Розничный блок", "Специалист", "гибрид", "2024-01-01", FIRST],
                [THIRD, "T02", "ИТ-платформа", "Специалист", "удалёнка", "2026-06-20", ""],
            ],
            "survey_responses": [response("R000001", FIRST, q_pay="", q_recommend="0")],
            "comments": [comment("C00001", FIRST)],
            "exits": [[SECOND, "2026-03-16", "собственное желание", "добровольный"]],
            "market_context": [[
                "Источник", "eNPS", "финансовый сектор", "2025", "-100", "п.п.",
                "ежегодный опрос, вопрос о рекомендации работодателя",
            ]],
        }

    def write_inputs(self):
        for name, rows in self.rows.items():
            with (self.raw / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, delimiter=";")
                writer.writerow(TABLES[name].columns)
                writer.writerows(rows)

    def audit(self):
        self.write_inputs()
        return run_audit(self.raw)

    def issues(self, audit, rule):
        return [item for item in audit.issues if item["rule"] == rule]

    def test_expected_features_do_not_hide_or_block(self):
        audit = self.audit()
        self.assertFalse(audit.critical())
        self.assertEqual(len(self.issues(audit, "missing_pay")), 1)
        self.assertEqual(len(self.issues(audit, "alternate_comment_date")), 1)
        self.assertEqual(audit.tables["comments"].loc[1, "text"], "Один; два")
        self.assertEqual(audit.profiles["survey_responses"]["scores"]["q_recommend"]["min"], 0)
        self.assertFalse(self.issues(audit, "benchmark_value"))

    def test_invalid_scores_and_impossible_comment_date(self):
        self.rows["survey_responses"].append(response(
            "R000002", FIRST, q_conditions="bad", q_growth="5.5", q_tools="11", q_pay="3.0",
            q_balance="0", q_recommend="10",
        ))
        self.rows["comments"].append(comment("C00002", FIRST, day="30.02.2026"))
        audit = self.audit()
        self.assertEqual(
            {item["evidence"]["field"] for item in self.issues(audit, "invalid_score")},
            {"q_conditions", "q_growth", "q_tools", "q_balance"},
        )
        self.assertEqual(self.issues(audit, "invalid_comment_date")[0]["record"], 2)
        self.assertFalse(audit.critical())

    def test_primary_keys_are_distinct_from_the_employee_wave_grain(self):
        self.rows["survey_responses"] += [
            response("R000002", FIRST, q_recommend="4"), response("R000002", SECOND),
            response("", SECOND, wave="2026-06", day="2026-06-15"),
        ]
        audit = self.audit()
        self.assertEqual([item["record"] for item in self.issues(audit, "duplicate_key")], [2, 3])
        self.assertEqual(self.issues(audit, "missing_key")[0]["record"], 4)
        self.assertEqual(self.issues(audit, "survey_pair_conflict")[0]["evidence"]["records"], [1, 2])
        self.assertTrue(audit.critical("survey_responses"))
        self.assertNotIn("respondents", audit.diagnostics["headcounts"]["2026-03"])

    def test_foreign_keys_and_comment_organization(self):
        self.rows["comments"] += [comment("C00002", UNKNOWN), comment("C00003", FIRST, team="T02")]
        self.rows["survey_responses"].append(response("R000002", "EMP99998"))
        audit = self.audit()
        self.assertEqual(self.issues(audit, "orphan_comment_employee")[0]["record"], 2)
        self.assertEqual(self.issues(audit, "comment_no_response")[0]["record"], 2)
        self.assertEqual(self.issues(audit, "orphan_employee")[0]["table"], "survey_responses")
        mismatch = self.issues(audit, "comment_org_mismatch")[0]
        self.assertEqual((mismatch["record"], mismatch["evidence"]["roster_value"]), (3, "T01"))

    def test_point_in_time_counts_and_distinct_respondents(self):
        self.rows["survey_responses"] += [
            response("R000002", FIRST, q_conditions="7.0", q_pay="", q_recommend="0.0"),
            response("R000003", SECOND),
            response("R000004", FIRST, wave="2026-06", day="2026-06-16"),
        ]
        audit = self.audit()
        waves = audit.diagnostics["headcounts"]
        self.assertEqual([wave["active"] for wave in waves.values()], [1, 1, 2])
        self.assertEqual([wave["respondents"] for wave in waves.values()], [1, 0, 0])
        self.assertEqual(waves["2026-03"]["on_exit_day"], 1)
        self.assertEqual(waves["2026-03"]["respondents_on_exit_day"], 1)
        self.assertEqual(self.issues(audit, "on_exit_day")[0]["record"], 3)
        self.assertEqual(self.issues(audit, "outside_wave_anchor")[0]["record"], 4)
        self.assertEqual(self.issues(audit, "survey_pair_duplicate")[0]["evidence"]["records"], [1, 2])
        self.assertFalse(audit.critical())

    def test_exit_window_and_employment_chronology(self):
        self.rows["exits"] += [
            [FIRST, "2025-09-30", "собственное желание", "добровольный"],
            [THIRD, "2025-08-31", "собственное желание", "добровольный"],
        ]
        self.rows["survey_responses"].append(response("R000002", THIRD))
        audit = self.audit()
        self.assertEqual(self.issues(audit, "exit_outside_12_months")[0]["record"], 2)
        self.assertEqual(self.issues(audit, "exit_out_of_window")[0]["record"], 3)
        self.assertEqual({item["table"] for item in self.issues(audit, "before_hire")}, {"exits", "survey_responses"})
        self.assertTrue(self.issues(audit, "after_exit"))
        self.assertNotIn("headcounts", audit.diagnostics)

    def test_swapped_wave_and_impossible_dates_are_critical(self):
        self.rows["survey_responses"] += [
            response("R000002", FIRST, day="2026-04-16"),
            response("R000003", FIRST, day="2026-03-32"),
        ]
        audit = self.audit()
        self.assertEqual(self.issues(audit, "outside_wave_month")[0]["record"], 2)
        self.assertEqual(self.issues(audit, "invalid_date")[0]["record"], 3)
        self.assertTrue(audit.critical("survey_responses"))

    def test_join_multiplies_same_wave_and_can_match_across_waves(self):
        self.rows["survey_responses"] += [
            response("R000002", FIRST), response("R000003", FIRST, "2026-06", "2026-06-15"),
        ]
        self.rows["comments"].append(comment("C00002", FIRST, text="Другой текст"))
        audit = self.audit()
        by_employee, by_pair = audit.diagnostics["joins"].values()
        self.assertEqual((by_employee["joined"], by_employee["cross_wave"]), (6, 2))
        self.assertEqual((by_pair["joined"], by_pair["cross_wave"]), (5, 0))
        self.assertEqual(self.issues(audit, "join_multiplier")[0]["evidence"]["joined_rows"], 4)

    def test_benchmark_metadata_is_checked(self):
        row = self.rows["market_context"][0].copy()
        row[3:] = ["2019", "111", "%", "методика не указана"]
        self.rows["market_context"].append(row)
        audit = self.audit()
        for rule in ("benchmark_period", "benchmark_value", "benchmark_method"):
            self.assertEqual(self.issues(audit, rule)[0]["record"], 2)

    def test_repeated_report_bytes_and_unchanged_csvs(self):
        audit = self.audit()
        before = {file.name: file.read_bytes() for file in self.raw.iterdir()}
        output_a, output_b = self.directory / "reports_a", self.directory / "reports_b"
        write_reports(audit, output_a)
        write_reports(run_audit(self.raw), output_b)
        for first in output_a.iterdir():
            self.assertEqual(first.read_bytes(), (output_b / first.name).read_bytes())
        self.assertEqual(before, {file.name: file.read_bytes() for file in self.raw.iterdir()})

    def test_cli_stops_on_bad_csv_and_still_writes_evidence(self):
        self.write_inputs()
        path = self.raw / "employees.csv"
        with path.open("a", encoding="utf-8") as stream:
            stream.write("EMP01111;T03\n")
        output = self.directory / "reports"
        result = subprocess.run(
            [sys.executable, "-m", "src.pulse_quality", "--data", str(self.raw), "--output", str(output)],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue((output / "data_quality.md").is_file())
        events = [json.loads(line) for line in (output / "data_quality_issues.jsonl").read_text(encoding="utf-8").splitlines()]
        malformed = next(item for item in events if item["rule"] == "input_unreadable")
        self.assertEqual((malformed["severity"], malformed["table"], malformed["record"]), ("critical", "employees", 4))


if __name__ == "__main__":
    unittest.main()
