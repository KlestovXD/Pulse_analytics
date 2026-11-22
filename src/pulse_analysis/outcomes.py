import numpy as np
import pandas as pd

from src.pulse_quality.contract import PAIR, SMALL_GROUP

from .metrics import SCORE_METRICS, Z95, wilson
from .prepare import attach


BOOTSTRAP_DRAWS = 1500
BOOTSTRAP_SEED = 2026
EVENT = "exit_within_90d"


def outcome_sample(datasets):
    cohort = datasets["exit_followup_90d"][0]
    survey = datasets["survey_employee_wave"][0]
    people = datasets["employees"][0]
    joined = attach(cohort, survey[[*PAIR, *SCORE_METRICS]], PAIR, "one_to_one")
    joined = attach(joined, people[["employee_id", "division", "grade", "work_format"]],
                    "employee_id", "many_to_one")
    joined["recommend_category"] = joined["recommend_signal"].map({
        -100: "Критики", 0: "Нейтральные", 100: "Сторонники",
    }).fillna("Без валидной оценки")
    return joined


def clustered_rate(frame):
    grouped = frame.groupby("employee_id")[EVENT].agg(["size", "sum"])
    if grouped.empty:
        return (float("nan"),) * 3
    counts = grouped["size"].to_numpy()
    events = grouped["sum"].to_numpy()
    estimate = events.sum() / counts.sum()
    if estimate in (0, 1):
        return wilson(int(estimate) * len(grouped), len(grouped))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(len(grouped), size=(BOOTSTRAP_DRAWS, len(grouped)))
    bootstrap = events[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return 100 * estimate, 100 * low, 100 * high


def risk_by_group(sample):
    cuts = [("total", "all", sample)]
    for column in ("wave", "division", "grade", "work_format", "recommend_category"):
        categories = sample[column].replace("", "Не указано")
        cuts.extend((column, name, frame) for name, frame in sample.groupby(categories, sort=True))

    rows = []
    for level, group, frame in cuts:
        n, people = len(frame), frame["employee_id"].nunique()
        events = int(frame[EVENT].sum())
        allowed = people >= SMALL_GROUP
        estimate = wilson(events, n) if level == "wave" else clustered_rate(frame)
        visible = lambda value: value if allowed else None
        rows.append({
            "dimension": level, "group": group, "observations": visible(n), "employees": visible(people),
            "exits": visible(events), "risk_pct": visible(estimate[0]), "lo": visible(estimate[1]),
            "hi": visible(estimate[2]), "status": "available" if allowed else "employees_lt_10",
        })
    return pd.DataFrame(rows)


def score_difference(data, metric):
    valid = data.loc[data[metric].notna()]
    event, score = valid[EVENT], valid[metric].astype(float)
    with_exit, without = score[event], score[~event]
    n_cases, n_controls = len(with_exit), len(without)
    visible = (
        valid.loc[event, "employee_id"].nunique() >= SMALL_GROUP
        and valid.loc[~event, "employee_id"].nunique() >= SMALL_GROUP
    )
    result = {"metric": metric, "status": "available" if visible else "valid_employees_lt_10"}
    values = dict.fromkeys(("n_exits", "n_no_exit", "mean_exit", "mean_no_exit", "difference", "lo", "hi"))
    if visible:
        mean_exit, mean_no_exit = with_exit.mean(), without.mean()
        diff = mean_exit - mean_no_exit
        influence = pd.concat(((with_exit - mean_exit) / n_cases, -(without - mean_no_exit) / n_controls))
        clustered = influence.groupby(valid["employee_id"]).sum()
        se = np.sqrt((clustered**2).sum() * len(clustered) / (len(clustered) - 1))
        values.update(n_exits=n_cases, n_no_exit=n_controls, mean_exit=mean_exit, mean_no_exit=mean_no_exit,
                      difference=diff, lo=diff - Z95 * se, hi=diff + Z95 * se)
    return result | values


def fit_logit(sample, metric, *, adjust_wave, unit):
    values = sample.loc[sample[metric].notna()].copy()
    y = values[EVENT].to_numpy(dtype=float)
    n = len(y)
    clusters, ids = pd.factorize(values["employee_id"], sort=True)
    event_clusters = values.loc[values[EVENT], "employee_id"].nunique()
    columns = [np.ones(n), (values[metric].to_numpy(dtype=float) - values[metric].mean()) / unit]
    if adjust_wave:
        columns.append(values["wave"].eq("2026-06").to_numpy(dtype=float))
    x = np.column_stack(columns)
    p = x.shape[1]
    base = {
        "metric": metric, "observations": n, "employees": len(ids), "exits": int(y.sum()),
        "employees_with_exit": event_clusters, "slopes": p - 1, "unit": unit, "wave_adjusted": adjust_wave,
        "odds_ratio": None, "lo": None, "hi": None,
    }
    # Запас событий применяется к уникальным сотрудникам. Модель с редкими исходами откладывается.
    if min(event_clusters, n - int(y.sum())) < 10 * (p - 1) or np.linalg.matrix_rank(x) < p:
        return base | {"status": "insufficient_information"}
    beta = np.zeros(p)
    for _ in range(50):
        probability = 1 / (1 + np.exp(-np.clip(x @ beta, -40, 40)))
        information = x.T @ ((probability * (1 - probability))[:, None] * x)
        if np.linalg.cond(information) > 1e10:
            break
        step = np.linalg.solve(information, x.T @ (y - probability))
        if np.max(np.abs(step)) < 1e-8:
            inverse = np.linalg.inv(information)
            cluster_scores = np.zeros((len(ids), p))
            np.add.at(cluster_scores, clusters, x * (y - probability)[:, None])
            adjustment = (len(ids) / (len(ids) - 1)) * ((n - 1) / (n - p))
            covariance = adjustment * inverse @ (cluster_scores.T @ cluster_scores) @ inverse
            se = np.sqrt(max(0, covariance[1, 1]))
            odds_ratio, low, high = np.exp([beta[1], beta[1] - Z95 * se, beta[1] + Z95 * se])
            return base | {"status": "converged", "odds_ratio": odds_ratio, "lo": low, "hi": high}
        beta += step
    return base | {"status": "no_stable_fit"}


def analyze_outcomes(datasets):
    sample = outcome_sample(datasets)
    risks = risk_by_group(sample)
    scores = pd.DataFrame(score_difference(sample, metric) for metric in SCORE_METRICS)
    models = [
        {"model": "engagement_adjusted", **fit_logit(sample, "engagement", adjust_wave=True, unit=10)},
        {"model": "engagement_same8", **fit_logit(sample.loc[sample["state8"].notna()], "engagement",
                                                adjust_wave=True, unit=10)},
        {"model": "state8_adjusted", **fit_logit(sample, "state8", adjust_wave=True, unit=10)},
        {"model": "recommend_adjusted", **fit_logit(sample, "recommend_signal", adjust_wave=True, unit=100)},
    ]
    for wave in ("2026-03", "2026-06"):
        frame = sample.loc[sample["wave"].eq(wave)]
        models.append({"model": f"engagement_{wave}", **fit_logit(frame, "engagement", adjust_wave=False, unit=10)})
    models = pd.DataFrame(models)
    for col in ("employees", "observations", "exits", "employees_with_exit"):
        models[col] = models[col].where(models["employees"].ge(SMALL_GROUP))
    return risks, scores, models
