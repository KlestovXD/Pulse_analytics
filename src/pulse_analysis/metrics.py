from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.pulse_quality.contract import EXIT_END, PAIR, SMALL_GROUP, TRAILING_YEAR_START, WAVE_DATES

from .prepare import THEMES, attach


Z95 = NormalDist().inv_cdf(0.975)
SCORE_METRICS = {
    "engagement": "Вовлечённость",
    "state8": "Состояние по восьми темам",
    "recommend_signal": "eNPS",
    "q_conditions": "Условия труда",
    "q_recognition": "Признание",
    "q_growth": "Развитие",
    "q_manager": "Руководитель",
    "q_pay": "Оплата",
    "q_balance": "Баланс",
    "q_goals": "Цели",
    "q_tools": "Инструменты",
}
DELTA_METRICS = ("participation", "engagement", "state8", "recommend_signal", *THEMES)
ORG_LEVELS = ("division", "team_id")


def mean_interval(values, bounds=None):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(values)
    if n == 0:
        return n, float("nan"), float("nan"), float("nan")
    mean = values.mean()
    if n < 2:
        return n, mean, float("nan"), float("nan")
    half = Z95 * values.std(ddof=1) / np.sqrt(n)
    low, high = mean - half, mean + half
    if bounds:
        low, high = max(bounds[0], low), min(bounds[1], high)
    return n, mean, low, high


def wilson(successes, total):
    if total == 0:
        return (float("nan"),) * 3
    p = successes / total
    denominator = 1 + Z95**2 / total
    center = p + Z95**2 / (2 * total)
    half = Z95 * np.sqrt(p * (1 - p) / total + Z95**2 / (4 * total**2))
    low = 0 if successes == 0 else 100 * (center - half) / denominator
    high = 100 if successes == total else 100 * (center + half) / denominator
    return 100 * p, low, high


def bounds_for(metric):
    if metric == "recommend_signal":
        return -100, 100
    return (1, 10) if metric in THEMES else (0, 100)


def group_org(frame):
    yield "total", "all", frame
    for level in ORG_LEVELS:
        for group, part in frame.groupby(level, sort=True):
            yield level, group, part


def wave_people(spine, at_wave):
    return attach(spine, at_wave[[*PAIR, *SCORE_METRICS]], PAIR, "one_to_one")


def put_estimate(row, name, estimate):
    value, low, high = estimate
    row.update({name: value, f"{name}_lo": low, f"{name}_hi": high})


def wave_metrics(people):
    rows = []
    for wave, members in people.groupby("wave", sort=True):
        for level, group, part in group_org(members):
            active, answered = int(part["active"].sum()), int(part["responded"].sum())
            active_inc = int(part["active_inclusive"].sum())
            answered_inc = int(part["responded_inclusive"].sum())
            allowed = answered >= SMALL_GROUP
            alt_allowed = allowed and answered_inc >= SMALL_GROUP
            selected = part.loc[part["responded"]]
            alternative = part.loc[part["responded_inclusive"]]
            row = {
                "wave": wave, "wave_date": WAVE_DATES[wave], "level": level, "group": group,
                "division": part["division"].iloc[0] if level == "team_id" else "",
                "active": active if allowed else None, "respondents": answered if allowed else None,
                "active_inclusive": active_inc if alt_allowed else None,
                "respondents_inclusive": answered_inc if alt_allowed else None,
                "publication_status": "available" if allowed else "respondents_lt_10",
            }
            put_estimate(row, "participation", wilson(answered, active) if allowed else (None,) * 3)
            row["participation_inclusive"] = wilson(answered_inc, active_inc)[0] if alt_allowed else None
            n8, *same_sample = mean_interval(selected.loc[selected["state8"].notna(), "engagement"], (0, 100))
            row["engagement_same8_n"] = n8 if n8 >= SMALL_GROUP else None
            put_estimate(row, "engagement_same8", same_sample if n8 >= SMALL_GROUP else (None,) * 3)
            for metric in SCORE_METRICS:
                n, value, low, high = mean_interval(selected[metric], bounds_for(metric))
                visible = allowed and n >= SMALL_GROUP
                row[f"{metric}_n"] = n if visible else None
                put_estimate(row, metric, (value, low, high) if visible else (None,) * 3)
                if metric in {"engagement", "state8", "recommend_signal"}:
                    alt = alternative[metric].dropna()
                    row[f"{metric}_inclusive"] = alt.mean() if alt_allowed and len(alt) >= SMALL_GROUP else None
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["level", "group", "wave"]).reset_index(drop=True)
    by_group = frame.groupby(["level", "group"], sort=False)
    for metric in ("active", "respondents", *DELTA_METRICS):
        previous = pd.to_numeric(by_group[metric].shift(), errors="coerce")
        current = pd.to_numeric(frame[metric], errors="coerce")
        frame[f"{metric}_delta"] = current - previous
    return frame


def paired_samples(people, earlier, later):
    cols = ["employee_id", *ORG_LEVELS, *SCORE_METRICS]
    old = people.loc[people["wave"].eq(earlier) & people["responded"], cols]
    new = people.loc[people["wave"].eq(later) & people["responded"], ["employee_id", *SCORE_METRICS]]
    return old.merge(new, on="employee_id", how="inner", suffixes=("_old", "_new"), validate="one_to_one")


def paired_metrics(people):
    rows = []
    waves = list(WAVE_DATES)
    repeated = people.loc[people["responded"]].groupby("employee_id")["wave"].nunique()
    all_three = repeated[repeated.eq(len(waves))].index
    for earlier, later in combinations(waves, 2):
        pairs = paired_samples(people, earlier, later)
        cohorts = {"pair": pairs}
        if earlier == waves[0] and later == waves[-1]:
            cohorts["all_three"] = pairs.loc[pairs["employee_id"].isin(all_three)]
        for cohort, sample in cohorts.items():
            for level, group, part in group_org(sample):
                for metric in SCORE_METRICS:
                    old, new = part[f"{metric}_old"], part[f"{metric}_new"]
                    n, value, low, high = mean_interval(new - old)
                    visible = n >= SMALL_GROUP
                    valid = old.notna() & new.notna()
                    rows.append({
                        "earlier": earlier, "later": later, "cohort": cohort,
                        "level": level, "group": group, "metric": metric,
                        "matched_respondents": len(part) if len(part) >= SMALL_GROUP else None,
                        "n": n if visible else None, "status": "available" if visible else "valid_pairs_lt_10",
                        "earlier_value": old[valid].mean() if visible else None,
                        "later_value": new[valid].mean() if visible else None,
                        "delta": value if visible else None,
                        "lo": low if visible else None, "hi": high if visible else None,
                    })
    return pd.DataFrame(rows)


def sample_composition(people):
    rows = []
    for wave, cohort in people.groupby("wave", sort=True):
        for dimension in ("division", "grade", "work_format"):
            categories = cohort[dimension].replace("", "Не указано")
            active_total, response_total = int(cohort["active"].sum()), int(cohort["responded"].sum())
            for group, part in cohort.groupby(categories, sort=True):
                active, responders = int(part["active"].sum()), int(part["responded"].sum())
                visible = responders >= SMALL_GROUP
                row = {
                    "wave": wave, "dimension": dimension, "group": group,
                    "active": active if visible else None, "respondents": responders if visible else None,
                    "status": "available" if visible else "respondents_lt_10",
                    "active_share": 100 * active / active_total if visible and active_total else None,
                    "respondent_share": 100 * responders / response_total if visible and response_total else None,
                }
                put_estimate(row, "participation", wilson(responders, active) if visible else (None,) * 3)
                rows.append(row)
    frame = pd.DataFrame(rows)
    for col in ("respondent_share", "active_share"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["respondent_share_change"] = frame.groupby(["dimension", "group"])["respondent_share"].diff()
    frame["representation_gap"] = frame["respondent_share"] - frame["active_share"]
    return frame


def decompose_engagement(people, paired):
    source = people.loc[people["responded"] & people["engagement"].notna()]
    cuts = source.groupby(["wave", "division"])["engagement"].agg(["size", "mean"])
    totals = source.groupby("wave")["engagement"].agg(["size", "mean"])
    panels = paired.loc[paired["level"].eq("total") & paired["cohort"].eq("pair")
                        & paired["metric"].eq("engagement")].set_index(["earlier", "later"])
    rows = []
    for earlier, later in combinations(WAVE_DATES, 2):
        row = dict.fromkeys(("observed_delta", "within_division", "division_mix", "paired_delta",
                             "earlier_selection", "later_selection"))
        row.update(earlier=earlier, later=later)
        if {earlier, later} <= set(totals.index) and totals.loc[[earlier, later], "size"].min() >= SMALL_GROUP:
            old_all, new_all = totals.loc[[earlier, later], "mean"]
            row["observed_delta"] = new_all - old_all
            panel = panels.loc[(earlier, later)]
            if pd.notna(panel["delta"]):
                row["paired_delta"] = panel["delta"]
                row["earlier_selection"] = panel["earlier_value"] - old_all
                row["later_selection"] = new_all - panel["later_value"]
        if earlier in cuts.index.get_level_values("wave") and later in cuts.index.get_level_values("wave"):
            first, last = cuts.loc[earlier], cuts.loc[later]
            if set(first.index) == set(last.index) and min(first["size"].min(), last["size"].min()) >= SMALL_GROUP:
                p0, p1 = first["size"] / first["size"].sum(), last["size"] / last["size"].sum()
                m0, m1 = first["mean"], last["mean"]
                row["within_division"] = ((p0 + p1) / 2 * (m1 - m0)).sum()
                row["division_mix"] = ((m0 + m1) / 2 * (p1 - p0)).sum()
        rows.append(row)
    return pd.DataFrame(rows)


def annual_turnover(lifecycle):
    start, end = pd.Timestamp(TRAILING_YEAR_START), pd.Timestamp(EXIT_END)
    end_exclusive = end + pd.Timedelta(days=1)
    days_in_window = (end_exclusive - start).days
    frame = lifecycle.copy()
    hired = frame["hire_date"].clip(lower=start)
    for suffix, shift in (("", 0), ("_inclusive", 1)):
        exit_date = frame["exit_date"] + pd.Timedelta(days=shift)
        stop = exit_date.fillna(end_exclusive).clip(upper=end_exclusive)
        frame[f"person_days{suffix}"] = (stop - hired).dt.days.clip(lower=0)
    frame["all_exits"] = frame["exit_date"].between(start, end)
    frame["voluntary_exits"] = frame["all_exits"] & frame["exit_type"].eq("добровольный")
    rows = []
    for level, group, part in group_org(frame):
        exits, voluntary = int(part["all_exits"].sum()), int(part["voluntary_exits"].sum())
        row = {"level": level, "group": group, "period_start": start.date(), "period_end": end.date(),
               "days_in_window": days_in_window, "all_exits": exits, "voluntary_exits": voluntary}
        for suffix in ("", "_inclusive"):
            exposure = int(part[f"person_days{suffix}"].sum())
            mean_size = exposure / days_in_window
            row[f"person_days{suffix}"] = exposure
            row[f"mean_headcount{suffix}"] = mean_size
            row[f"turnover{suffix}"] = 100 * exits / mean_size if mean_size else None
            row[f"voluntary_turnover{suffix}"] = 100 * voluntary / mean_size if mean_size else None
        rows.append(row)
    return pd.DataFrame(rows)


def team_metrics(waves, paired, annual):
    team = waves.loc[waves["level"].eq("team_id")].copy()
    keys = ["group", "wave"]
    previous_wave = dict(zip(list(WAVE_DATES)[1:], list(WAVE_DATES)[:-1]))
    shifts = paired.loc[
        paired["level"].eq("team_id") & paired["cohort"].eq("pair") & paired["metric"].eq("engagement")
        & paired["earlier"].eq(paired["later"].map(previous_wave)),
        ["group", "later", "n", "delta", "lo", "hi"],
    ].rename(columns={"later": "wave", **{c: f"paired_engagement_{c}" for c in ("n", "delta", "lo", "hi")}})
    team = attach(team, shifts, keys, "one_to_one")
    date_last_wave = list(WAVE_DATES)[-1]
    annual_fields = ("period_start", "period_end", "all_exits", "voluntary_exits",
                     "mean_headcount", "turnover", "voluntary_turnover", "turnover_inclusive",
                     "voluntary_turnover_inclusive")
    by_team = annual.loc[annual["level"].eq("team_id"), ["group", *annual_fields]]
    by_team = by_team.rename(columns={c: f"annual_{c}" for c in annual_fields}).assign(wave=date_last_wave)
    team = attach(team, by_team, keys, "one_to_one")
    visible = team["publication_status"].eq("available")
    for col in annual_fields:
        team[f"annual_{col}"] = team[f"annual_{col}"].where(visible)
    team["attention"] = "no_paired_decline_signal"
    team.loc[team["paired_engagement_n"].isna(), "attention"] = "insufficient_paired_data"
    team.loc[team["wave"].eq(list(WAVE_DATES)[0]), "attention"] = "no_previous_wave"
    team.loc[team["paired_engagement_hi"].lt(0), "attention"] = "paired_decline_signal"
    team.loc[~visible, "attention"] = "suppressed"
    return team.rename(columns={"group": "team_id"}).drop(columns="level")
