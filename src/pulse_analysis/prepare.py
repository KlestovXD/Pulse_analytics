from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.pulse_quality.audit import run_audit
from src.pulse_quality.contract import EXIT_END, PAIR, SCORES, TRAILING_YEAR_START, WAVE_DATES


THEMES = SCORES[:-1]
ENGAGEMENT = ("q_recognition", "q_growth", "q_manager", "q_goals")
GROUPS = ("team_id", "division", "grade", "work_format")
HORIZON = 90


def active_at(frame, day, *, inclusive_exit=False):
    exit_date = frame["exit_date"]
    still_employed = exit_date.ge(day) if inclusive_exit else exit_date.gt(day)
    return frame["hire_date"].le(day) & (exit_date.isna() | still_employed)


def check_key(frame, fields):
    key = frame[list(fields)]
    if key.isna().any().any() or key.eq("").any().any() or frame.duplicated(list(fields)).any():
        raise ValueError(f"Missing or duplicate analytical key {fields}")


def attach(left, right, key, validation):
    merged = left.merge(right, on=key, how="left", validate=validation)
    if len(merged) != len(left):
        raise ValueError(f"Unexpected row count after merge on {key}")
    return merged


def add_scores(frame):
    """Индексы используют полные наборы вопросов с одинаковыми весами."""
    result = frame.copy()
    for name, questions in (("engagement", ENGAGEMENT), ("state8", THEMES)):
        result[name] = (result[list(questions)].mean(axis=1, skipna=False) - 1) * (100 / 9)
    recommend = result["q_recommend"]
    result["recommend_signal"] = recommend.where(recommend.isna(), 0).astype("Float64")
    result.loc[recommend.ge(9).fillna(False), "recommend_signal"] = 100
    result.loc[recommend.le(6).fillna(False), "recommend_signal"] = -100
    return result


def survey_versions(audit):
    source = audit.tables["survey_responses"]
    versions = source[["response_id", *PAIR]].assign(
        source_record=source.index,
        submitted_at_raw=source["submitted_at"],
        submitted_at=audit.dates["survey_responses"],
    )
    cutoff = versions["wave"].map(WAVE_DATES).astype("datetime64[us]")
    versions["received_by_cutoff"] = versions["submitted_at"].le(cutoff)
    for col in SCORES:
        raw = source[col]
        number = pd.to_numeric(raw, errors="coerce")
        in_range = number.between(0 if col == "q_recommend" else 1, 10)
        valid = number.where(in_range).mod(1).eq(0).fillna(False)
        versions[col] = number.where(valid).astype("Int64")
        versions[f"{col}_input_state"] = raw.eq("").map({True: "missing", False: "invalid"})
        versions.loc[valid, f"{col}_input_state"] = "valid"
    return versions


def collapse_survey(versions):
    grouped = versions.groupby(PAIR, sort=True)
    result = grouped.agg(
        first_submitted_at=("submitted_at", "min"),
        as_of=("submitted_at", "max"),
        source_response_count=("response_id", "size"),
    )
    for col in SCORES:
        distinct = grouped[col].nunique()
        has_invalid = versions[f"{col}_input_state"].eq("invalid").groupby(
            [versions[key] for key in PAIR], sort=True
        ).any()
        result[col] = grouped[col].first().where(distinct.eq(1)).astype("Int64")
        state = pd.Series("missing", index=result.index, dtype="string")
        state.loc[has_invalid] = "invalid"
        state.loc[distinct.eq(1)] = "resolved"
        state.loc[distinct.gt(1)] = "conflict"
        result[f"{col}_state"] = state
    return add_scores(result.reset_index())


def workforce(lifecycle, at_wave):
    waves = pd.DataFrame({
        "wave": list(WAVE_DATES),
        "wave_date": pd.to_datetime(list(WAVE_DATES.values())),
    })
    spine = lifecycle[["employee_id", *GROUPS, "hire_date", "exit_date"]].merge(waves, how="cross")
    spine["active"] = active_at(spine, spine["wave_date"])
    spine["active_inclusive"] = active_at(spine, spine["wave_date"], inclusive_exit=True)
    spine = attach(spine.drop(columns=["hire_date", "exit_date"]),
                   at_wave[PAIR].assign(response_received=True), PAIR, "one_to_one")
    spine["response_received"] = spine["response_received"].eq(True)
    spine["responded"] = spine["active"] & spine["response_received"]
    spine["responded_inclusive"] = spine["active_inclusive"] & spine["response_received"]
    return spine


def prepare_comments(audit):
    source = audit.tables["comments"]
    frame = source.assign(source_record=source.index, submitted_at_raw=source["submitted_at"],
                          submitted_at=audit.dates["comments"])
    rules = defaultdict(set)
    for issue in audit.issues:
        if issue["table"] == "comments" and issue["grain"] == "record":
            rules[issue["record"]].add(issue["rule"])
    frame["quality_rules"] = ["|".join(sorted(rules[i])) for i in frame.index]
    excluded = {
        "orphan_comment_employee", "comment_no_response", "comment_org_mismatch", "invalid_comment_date",
        "comment_outside_wave_month", "comment_outside_employment", "on_exit_day", "comment_placeholder",
        "invalid_optional_id", "invalid_enum", "missing_field",
    }
    frame["usable_team_text"] = [not (rules[i] & excluded) for i in frame.index]
    return frame


def followup(survey, lifecycle):
    # Событие присоединяется только в таблице исходов. В признаках его даты нет.
    cohort = attach(survey[[*PAIR, "as_of"]], lifecycle[["employee_id", "hire_date", "exit_date"]],
                    "employee_id", "many_to_one")
    cohort["horizon_end"] = cohort["as_of"] + pd.Timedelta(days=HORIZON)
    included = (
        cohort["wave"].isin(("2026-03", "2026-06"))
        & cohort["horizon_end"].le(pd.Timestamp(EXIT_END))
        & active_at(cohort, cohort["as_of"])
    )
    cohort = cohort.loc[included].copy()
    delay = (cohort["exit_date"] - cohort["as_of"]).dt.days
    cohort["exit_within_90d"] = delay.between(1, HORIZON)
    return cohort[[*PAIR, "as_of", "horizon_end", "exit_within_90d"]].reset_index(drop=True)


def prepare(raw: Path):
    audit = run_audit(raw)
    if audit.critical():
        failed = sorted({f"{i['table']}/{i['rule']}" for i in audit.issues if i["severity"] == "critical"})
        raise ValueError("Critical audit checks: " + ", ".join(failed))
    e = audit.tables["employees"]
    employees = e.assign(source_record=e.index, hire_date=audit.dates["employees"])
    x = audit.tables["exits"]
    exits = x.assign(source_record=x.index, exit_date=audit.dates["exits"])
    exits["in_annual_window"] = exits["exit_date"].between(pd.Timestamp(TRAILING_YEAR_START), pd.Timestamp(EXIT_END))
    lifecycle = attach(employees, exits[["employee_id", "exit_date", "exit_type"]], "employee_id", "one_to_one")

    versions = survey_versions(audit)
    survey = collapse_survey(versions)
    timely_versions = versions.loc[versions["received_by_cutoff"]]
    at_wave = collapse_survey(timely_versions)
    spine = workforce(lifecycle, at_wave)
    outcome = followup(survey, lifecycle)
    comments = prepare_comments(audit)
    expected_rows = (
        (employees, len(e)), (exits, len(x)), (comments, len(audit.tables["comments"])),
        (versions, len(audit.tables["survey_responses"])),
        (survey, len(versions[PAIR].drop_duplicates())),
        (at_wave, len(timely_versions[PAIR].drop_duplicates())),
        (spine, len(employees) * len(WAVE_DATES)),
    )
    if any(len(frame) != count for frame, count in expected_rows):
        raise ValueError("Unexpected analytical row count")
    if (survey["source_response_count"].sum() != len(versions)
            or at_wave["source_response_count"].sum() != len(timely_versions)):
        raise ValueError("Survey lineage does not reconcile")
    datasets = {
        "employees": (employees, ("employee_id",)),
        "exits": (exits, ("employee_id",)),
        "survey_lineage": (versions, ("response_id",)),
        "survey_employee_wave": (survey, tuple(PAIR)),
        "survey_at_wave": (at_wave, tuple(PAIR)),
        "employee_wave": (spine, tuple(PAIR)),
        "comments": (comments, ("comment_id",)),
        "exit_followup_90d": (outcome, tuple(PAIR)),
    }
    for frame, key in datasets.values():
        check_key(frame, key)
    return audit, lifecycle, datasets
