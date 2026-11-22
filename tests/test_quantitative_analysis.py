import csv
import hashlib
import unittest
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import pandas as pd

from src.pulse_analysis.__main__ import calculate, publish
from src.pulse_analysis.metrics import annual_turnover, mean_interval, wave_metrics, wilson
from src.pulse_analysis.outcomes import fit_logit
from src.pulse_analysis.prepare import (
    add_scores, attach, check_key, collapse_survey, followup, workforce,
)
from src.pulse_quality.contract import PAIR, SCORES


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def dates(frame, *columns):
    for col in columns:
        frame[col] = pd.to_datetime(frame[col])
    return frame


def raw_csv(name):
    with (RAW / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


class MetricTests(unittest.TestCase):
    def test_index_endpoints_and_missing_component(self):
        rows = [dict.fromkeys(SCORES, 1), dict.fromkeys(SCORES, 10), dict.fromkeys(SCORES, 5)]
        rows[2].update(q_pay=None, q_recommend=None)
        result = add_scores(pd.DataFrame(rows).astype("Int64"))
        self.assertEqual(result["engagement"].iloc[:2].tolist(), [0, 100])
        self.assertAlmostEqual(result.loc[2, "engagement"], 400 / 9)
        self.assertTrue(pd.isna(result.loc[2, "state8"]))
        self.assertTrue(pd.isna(result.loc[2, "recommend_signal"]))

    def test_enps_neutrals_and_valid_denominator(self):
        rows = [dict.fromkeys(SCORES, 5) | {"q_recommend": i} for i in (0, 6, 7, 8, 9, 10, 10, None)]
        signals = add_scores(pd.DataFrame(rows).astype("Int64"))["recommend_signal"]
        self.assertEqual(signals.iloc[:-1].tolist(), [-100, -100, 0, 0, 100, 100, 100])
        self.assertAlmostEqual(signals.mean(), 100 / 7)

    def test_repeated_answers_keep_consensus_and_flag_conflicts(self):
        base = {col: 7 for col in SCORES} | {f"{col}_input_state": "valid" for col in SCORES}
        common = {"employee_id": "E1", "wave": "2026-03"}
        first = base | common | {"response_id": "1", "submitted_at": "2026-03-16"}
        second = base | common | {
            "response_id": "2", "submitted_at": "2026-03-17",
            "q_conditions": 5, "q_pay": None, "q_pay_input_state": "missing",
            "q_tools": None, "q_tools_input_state": "invalid",
        }
        versions = dates(pd.DataFrame([first, second]), "submitted_at")
        both = collapse_survey(versions).iloc[0]
        self.assertTrue(pd.isna(both["q_conditions"]))
        self.assertEqual(both["q_conditions_state"], "conflict")
        self.assertEqual(both["q_pay"], 7)
        self.assertEqual(both["q_tools"], 7)
        self.assertEqual(both["source_response_count"], 2)
        self.assertEqual(both["as_of"].date(), date(2026, 3, 17))
        cutoff = collapse_survey(versions.loc[versions["submitted_at"].le("2026-03-16")]).iloc[0]
        self.assertEqual(cutoff["q_conditions"], 7)
        self.assertEqual(cutoff["source_response_count"], 1)
        self.assertTrue(pd.isna(both["state8"]))
        self.assertAlmostEqual(both["engagement"], 600 / 9)

    def test_active_denominator_and_duplicate_merge(self):
        lifecycle = pd.DataFrame({
            "employee_id": ["E1", "E2", "E3"], "team_id": ["T01"] * 3, "division": ["IT"] * 3,
            "work_format": ["office"] * 3, "grade": ["g"] * 3,
            "hire_date": ["2026-03-16", "2025-01-01", "2026-03-17"],
            "exit_date": [None, "2026-03-16", None],
        })
        lifecycle = dates(lifecycle, "hire_date", "exit_date")
        survey = pd.DataFrame({"employee_id": ["E1", "E2", "E3"], "wave": ["2026-03"] * 3})
        march = workforce(lifecycle, survey).query("wave == '2026-03'")
        self.assertEqual(march["active"].sum(), 1)
        self.assertEqual(march["responded"].sum(), 1)
        self.assertEqual(march["active_inclusive"].sum(), 2)
        self.assertEqual(march["responded_inclusive"].sum(), 2)
        with self.assertRaises(pd.errors.MergeError):
            workforce(lifecycle, pd.concat([survey, survey.iloc[:1]]))
        with self.assertRaises(ValueError):
            check_key(pd.concat([survey, survey.iloc[:1]]), PAIR)
        self.assertEqual(len(attach(survey, lifecycle[["employee_id", "team_id"]], "employee_id", "many_to_one")), 3)

    def test_followup_temporal_boundaries(self):
        lifecycle = dates(pd.DataFrame({
            "employee_id": ["day0", "day1", "day90", "day91", "sept"],
            "hire_date": ["2025-01-01"] * 5,
            "exit_date": ["2026-03-16", "2026-03-17", "2026-06-14", "2026-06-15", "2026-09-15"],
        }), "hire_date", "exit_date")
        survey = dates(pd.DataFrame({
            "employee_id": lifecycle["employee_id"],
            "as_of": ["2026-03-16"] * 4 + ["2026-09-14"],
            "wave": ["2026-03"] * 4 + ["2026-09"],
        }), "as_of")
        observed = followup(survey, lifecycle).set_index("employee_id")
        self.assertEqual(observed["exit_within_90d"].to_dict(), {"day1": True, "day90": True, "day91": False})
        self.assertNotIn("exit_date", observed)

    def test_annual_exposure_and_calendar_window(self):
        lifecycle = dates(pd.DataFrame({
            "employee_id": ["exit_on_start", "full", "prior", "hire_on_end"],
            "team_id": ["T01"] * 4, "division": ["IT"] * 4,
            "hire_date": ["2020-01-01"] * 3 + ["2026-09-30"],
            "exit_date": ["2025-10-01", None, "2025-09-30", None],
            "exit_type": ["добровольный", None, "вынужденный", None],
        }), "hire_date", "exit_date")
        total = annual_turnover(lifecycle).query("level == 'total'").iloc[0]
        self.assertEqual(total["all_exits"], 1)
        self.assertEqual(total["voluntary_exits"], 1)
        self.assertEqual(total["person_days"], 366)
        self.assertEqual(total["person_days_inclusive"], 367)
        self.assertAlmostEqual(total["turnover"], 36500 / 366)

    def test_empty_groups_and_small_sample_disclosure(self):
        self.assertTrue(all(pd.isna(v) for v in wilson(0, 0)))
        self.assertEqual(mean_interval([])[0], 0)
        rate, low, high = wilson(0, 10)
        self.assertEqual(rate, 0)
        self.assertEqual(low, 0)
        self.assertGreater(high, 0)
        columns = ["active", "responded", "active_inclusive", "responded_inclusive"]
        frame = add_scores(pd.DataFrame([dict.fromkeys(SCORES, 7)]))
        frame = frame.assign(employee_id="E1", wave="2026-03", team_id="T01", division="Дивизион")
        frame[columns] = True
        metric = wave_metrics(frame).query("level == 'team_id'").iloc[0]
        self.assertEqual(metric["publication_status"], "respondents_lt_10")
        for col in ("respondents", "active", "engagement", "engagement_n", "engagement_delta", "participation_lo"):
            self.assertTrue(pd.isna(metric[col]), col)

    def test_logistic_recovers_two_group_odds_ratio(self):
        sample = pd.DataFrame({
            "employee_id": range(80), "score": [0] * 40 + [1] * 40,
            "exit_within_90d": [True] * 10 + [False] * 30 + [True] * 20 + [False] * 20,
        })
        result = fit_logit(sample, "score", adjust_wave=False, unit=1)
        self.assertEqual(result["status"], "converged")
        self.assertAlmostEqual(result["odds_ratio"], 3, places=6)
        self.assertLess(result["lo"], 3)
        self.assertGreater(result["hi"], 3)
        self.assertEqual(fit_logit(sample.iloc[:5], "score", adjust_wave=False, unit=1)["status"],
                         "insufficient_information")

    def test_pipeline_cannot_target_raw_directory(self):
        before = hashlib.sha256((RAW / "employees.csv").read_bytes()).digest()
        with self.assertRaises(ValueError):
            publish(RAW, RAW, ROOT / "reports" / "quantitative_analysis.md")
        self.assertEqual(before, hashlib.sha256((RAW / "employees.csv").read_bytes()).digest())


class IndependentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets, cls.checksums = calculate(RAW)

    def test_wave_metrics_and_traceable_grains_match_raw_csv_control(self):
        employees, exits, responses, comments = map(raw_csv, ("employees", "exits", "survey_responses", "comments"))
        hire = {r["employee_id"]: r["hire_date"] for r in employees}
        exit_date = {r["employee_id"]: r["exit_date"] for r in exits}
        grouped = defaultdict(list)
        for r in responses:
            grouped[r["employee_id"], r["wave"]].append(r)
        self.assertEqual(len(self.datasets["survey_lineage"][0]), len(responses))
        self.assertEqual(len(self.datasets["survey_employee_wave"][0]), len(grouped))
        self.assertEqual(len(self.datasets["comments"][0]), len(comments))
        metrics = self.datasets["wave_metrics"][0].query("level == 'total'").set_index("wave")
        for wave, day in (("2026-03", "2026-03-16"), ("2026-06", "2026-06-15"), ("2026-09", "2026-09-14")):
            active = {i for i in hire if hire[i] <= day < exit_date.get(i, "9999-12-31")}
            eng, rec = [], []
            observed = 0
            for i in active:
                usable = [r for r in grouped[i, wave] if r["submitted_at"] <= day]
                if not usable:
                    continue
                observed += 1
                def consensus(col):
                    distinct = {float(r[col]) for r in usable if r[col]}
                    return next(iter(distinct)) if len(distinct) == 1 else None
                local = [consensus(c) for c in ("q_recognition", "q_growth", "q_manager", "q_goals")]
                if None not in local:
                    eng.append((mean(local) - 1) / 9 * 100)
                val = consensus("q_recommend")
                if val is not None:
                    rec.append(100 * ((val >= 9) - (val <= 6)))
            self.assertEqual(metrics.loc[wave, "active"], len(active))
            self.assertEqual(metrics.loc[wave, "respondents"], observed)
            self.assertAlmostEqual(metrics.loc[wave, "participation"], observed / len(active) * 100)
            self.assertAlmostEqual(metrics.loc[wave, "engagement"], mean(eng))
            self.assertAlmostEqual(metrics.loc[wave, "recommend_signal"], mean(rec))

    def test_annual_turnover_and_followup_match_calendar_control(self):
        employees, exits, responses = map(raw_csv, ("employees", "exits", "survey_responses"))
        departure = {r["employee_id"]: date.fromisoformat(r["exit_date"]) for r in exits}
        hire = {r["employee_id"]: date.fromisoformat(r["hire_date"]) for r in employees}
        start, end = date(2025, 10, 1), date(2026, 9, 30)
        calendar = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        exposure = sum(hire[employee] <= day < departure.get(employee, date.max) for day in calendar for employee in hire)
        expected_events = Counter(r["exit_type"] for r in exits if start <= date.fromisoformat(r["exit_date"]) <= end)
        annual = self.datasets["annual_turnover"][0].query("level == 'total'").iloc[0]
        self.assertEqual(annual["person_days"], exposure)
        self.assertEqual(annual["all_exits"], expected_events.total())
        self.assertAlmostEqual(annual["voluntary_turnover"], 100 * expected_events["добровольный"] / (exposure / len(calendar)))

        latest = defaultdict(lambda: date.min)
        for row in responses:
            key = (row["employee_id"], row["wave"])
            latest[key] = max(latest[key], date.fromisoformat(row["submitted_at"]))
        expected = {
            (employee, wave): day < departure.get(employee, date.max) <= day + timedelta(days=90)
            for (employee, wave), day in latest.items()
            if wave in {"2026-03", "2026-06"} and hire[employee] <= day < departure.get(employee, date.max)
            and day + timedelta(days=90) <= end
        }
        actual = self.datasets["exit_followup_90d"][0].set_index(PAIR)["exit_within_90d"].to_dict()
        self.assertEqual(actual, expected)

    def test_decompositions_close_and_publication_threshold_applies(self):
        frame = self.datasets["engagement_decomposition"][0]
        for row in frame.itertuples():
            self.assertAlmostEqual(row.observed_delta, row.within_division + row.division_mix)
            self.assertAlmostEqual(row.observed_delta, row.paired_delta + row.earlier_selection + row.later_selection)
        team = self.datasets["team_metrics"][0]
        suppressed = team.loc[team["publication_status"].eq("respondents_lt_10")]
        self.assertFalse(suppressed.select_dtypes("number").notna().any().any())
        self.assertTrue(team.loc[team["wave"].ne("2026-09"), "annual_turnover"].isna().all())
        self.assertEqual(len(self.datasets["market_comparability"][0]), len(raw_csv("market_context")))

    def test_market_comparison_preserves_scope_and_calculates_only_admitted_differences(self):
        rows = self.datasets["market_comparison"][0].set_index("comparison_id")
        synthetic = rows.loc[rows["source_type"].eq("synthetic")]
        self.assertEqual(len(synthetic), len(raw_csv("market_context")))
        self.assertEqual(synthetic["absolute_difference"].notna().sum(), 1)
        self.assertTrue(synthetic["source_url"].eq("").all())
        self.assertTrue(rows.loc[rows["status"].eq("несопоставимые"), "absolute_difference"].isna().all())
        self.assertTrue(rows.loc[rows["publication_timing"].eq("after_cutoff"), "absolute_difference"].isna().all())
        self.assertEqual(rows.loc["nra25-turn", "publication_timing"], "after_cutoff")
        exposure = 336612 / 365
        rate = 100 * 133 / exposure
        self.assertAlmostEqual(rows.loc["synthetic-01", "absolute_difference"], rate - 12.5)
        self.assertAlmostEqual(rows.loc["domrf25-total", "relative_difference_pct"], 100 * (rate - 21.1) / 21.1)
        self.assertAlmostEqual(rows.loc["rosstat24-share", "absolute_difference"], 100 * 121 / 133 - 75.2)


if __name__ == "__main__":
    unittest.main()
