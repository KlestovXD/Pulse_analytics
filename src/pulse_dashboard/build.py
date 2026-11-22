import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

from src.pulse_comments.analytics import POLARITY_TOPICS
from src.pulse_comments.prepare import read_csv
from src.pulse_comments.schema import SCHEMA
from src.pulse_quality.contract import SMALL_GROUP
from src.pulse_quality.report import write_text


ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "dashboard"
METRICS = ("participation", "engagement", "recommend_signal")
TONES = SCHEMA["properties"]["sentiment"]["enum"]
TOPICS = ("оплата", "нагрузка", "признание", "развитие", "руководитель",
          "адаптация", "инструменты", "условия", "цели")
Z95 = NormalDist().inv_cdf(0.975)


def rows(path, columns):
    return read_csv(path, columns)[0]


def benjamini_hochberg(p_values):
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted, running = {}, 1.0
    for rank in range(len(ordered), 0, -1):
        key, probability = ordered[rank - 1]
        running = min(running, probability * len(ordered) / rank)
        adjusted[key] = running
    return adjusted


def intervals_to_signals(pairs, waves):
    previous = dict(zip(waves[1:], waves[:-1]))
    candidate = defaultdict(dict)
    for row in pairs:
        later, earlier = row["later"], row["earlier"]
        if (row["level"] != "team_id" or row["cohort"] != "pair" or row["metric"] != "engagement"
                or previous.get(later) != earlier or row["status"] != "available"):
            continue
        delta, lo, hi = map(float, (row["delta"], row["lo"], row["hi"]))
        n = int(row["n"])
        width = (hi - lo) / 2
        if n < SMALL_GROUP or width <= 0:
            # A normal interval with no estimable variance does not supply a valid test.
            continue
        p = math.erfc(abs(delta) * Z95 / width / math.sqrt(2))
        candidate[later][row["group"]] = {
            "team": row["group"], "wave": later, "previous": earlier, "delta": delta, "lo": lo, "hi": hi, "n": n, "p": p,
        }
    signals, families = [], {}
    for later in waves[1:]:
        group = candidate[later]
        families[later] = len(group)
        adjusted = benjamini_hochberg({key: values["p"] for key, values in group.items()})
        for team, row in group.items():
            if row["hi"] < 0:
                q = adjusted[team]
                signals.append({key: value for key, value in row.items() if key != "p"} |
                               {"q": round(q, 3), "confirmed": q <= 0.05})
    return signals, families


def protected_teams(team_rows):
    """Use one stable protection set across the waves, with two hidden siblings per affected parent."""
    members, hidden = defaultdict(set), set()
    for row in team_rows:
        team = row["team_id"]
        members[row["division"]].add(team)
        eligible = (
            row["publication_status"] == "available" and bool(row["respondents"])
            and int(row["respondents"]) >= SMALL_GROUP
            and all(row[metric] for metric in METRICS)
            and all(row[f"{m}_n"] and int(row[f"{m}_n"]) >= SMALL_GROUP for m in METRICS[1:])
        )
        if not eligible:
            hidden.add(team)
    for division, children in members.items():
        if len(children & hidden) == 1:
            complement = sorted(children - hidden)
            if not complement:
                raise ValueError(f"No complementary team for the protected division {division}")
            hidden.add(complement[0])
    return hidden


def metrics(row, columns=METRICS):
    return {
        metric: None if not row[metric] else [float(row[metric]), float(row[f"{metric}_lo"]), float(row[f"{metric}_hi"])]
        for metric in columns
    }


def protect_comment_grid(total, groups, *, exclusive=False):
    """Hide complements across divisions and, for tone, within each displayed row."""
    changed = True
    while changed:
        changed = False
        for key in total:
            if total[key] is not None and sum(group[key] is None for group in groups) == 1:
                visible = [group for group in groups if group[key] is not None]
                if visible:
                    min(visible, key=lambda group: (group[key] == 0, group[key]))[key] = None
                else:
                    total[key] = None
                changed = True
        if exclusive:
            for group in (total, *groups):
                if sum(value is None for value in group.values()) == 1:
                    other = min((key for key in group if group[key] is not None),
                                key=lambda key: (group[key] == 0, group[key]), default=None)
                    if other is not None:
                        group[other] = None
                        changed = True


def comment_snapshot(aggregate_root, waves, divisions):
    dist = rows(aggregate_root / "comment_distribution.csv",
                ("level", "wave", "division", "dimension", "category", "share", "status"))
    volumes = rows(aggregate_root / "comment_volume.csv",
                   ("level", "wave", "division", "substantive_comments", "status"))
    volume = {
        (r["wave"], r["division"]): (int(r["substantive_comments"]) if r["status"] == "available" else None)
        for r in volumes if r["level"] in {"wave", "division_wave"}
    }
    cells = {
        (r["wave"], r["division"], r["dimension"], r["category"]):
            round(100 * float(r["share"]), 1) if r["status"] == "available" and r["share"] else None
        for r in dist if r["level"] in {"wave", "division_wave"}
    }
    result = {}
    dimensions = (
        ("sentiment", TONES, "tone"),
        (POLARITY_TOPICS["положительная"], TOPICS, "positive_topics"),
        (POLARITY_TOPICS["отрицательная"], TOPICS, "negative_topics"),
    )
    if not {dimension for dimension, _, _ in dimensions}.issubset({r["dimension"] for r in dist}):
        raise ValueError("Rebuild the comment distributions with the local pulse_comments analyze command")
    for wave in waves:
        scopes = ("", *divisions)
        result[wave] = {scope: {"n": volume.get((wave, scope))} for scope in scopes}
        for dimension, values, field in dimensions:
            polarity = next((tone for tone, topic_dimension in POLARITY_TOPICS.items() if topic_dimension == dimension), None)
            for scope in scopes:
                show = polarity is None or result[wave][scope]["tone"][polarity] is not None
                result[wave][scope][field] = {
                    name: cells.get((wave, scope, dimension, name)) if show else None for name in values
                }
            protect_comment_grid(result[wave][""][field], [result[wave][d][field] for d in divisions],
                                 exclusive=field == "tone")
    comparison = rows(aggregate_root / "topic_survey_links.csv",
                      ("wave", "topic", "comparison", "difference", "status"))
    differences = {
        wave: {
            r["topic"]: round(float(r["difference"]), 2)
            for r in comparison if r["wave"] == wave and r["comparison"] == "problem_comment" and r["status"] == "available"
        } for wave in waves
    }
    return result, differences


def market_snapshot(processed, annual):
    rates = {"текучесть": "value", "текучесть добровольная": "voluntary"}
    banks = ("domrf25", "vtb25")
    comparison = rows(processed / "market_comparison.csv", (
        "source_id", "source", "source_url", "source_type", "metric", "value", "unit", "status",
        "period_start", "period_end", "published", "publication_timing", "absolute_difference",
        "internal_value", "internal_period_start", "internal_period_end",
    ))
    peers = {key: {"rates": {}} for key in banks}
    for row in comparison:
        if row["source_id"] not in peers:
            continue
        bank = peers[row["source_id"]]
        rate = rates.get(row["metric"])
        if (rate is None or rate in bank["rates"]
                or (row["period_start"], row["period_end"]) != ("2025-01-01", "2025-12-31")
                or (row["internal_period_start"], row["internal_period_end"]) != (annual["start"], annual["end"])
                or row["source_type"] != "bank_disclosure" or row["unit"] != "%"
                or row["publication_timing"] != "by_cutoff" or row["status"] != "условно сопоставимые"
                or not row["source_url"].startswith("https://")):
            raise ValueError(f"Unexpected market evidence: {row['source_id']}")
        internal = annual[rate][""]
        value = float(row["value"])
        gap = float(row["absolute_difference"])
        if not (math.isclose(float(row["internal_value"]), internal, abs_tol=1e-7)
                and math.isclose(gap, internal - value, abs_tol=1e-7)):
            raise ValueError(f"Market comparison is out of date: {row['source_id']}")
        bank.update(name=row["source"], url=row["source_url"], published=row["published"])
        bank["rates"][rate] = value
    if any(set(bank["rates"]) != set(rates.values()) for bank in peers.values()):
        raise ValueError("Annual official market comparators are incomplete")
    return list(peers.values())


def build_snapshot(processed):
    fields = (*METRICS, *(f"{m}_{x}" for m in METRICS for x in ("lo", "hi")))
    survey = rows(processed / "wave_metrics.csv", ("wave", "wave_date", "level", "group", "publication_status", *fields))
    wave_dates = {r["wave"]: r["wave_date"] for r in survey if r["level"] == "total"}
    waves = sorted(wave_dates)
    divisions = sorted({r["group"] for r in survey if r["level"] == "division"})
    survey_data = {scope: {} for scope in ("", *divisions)}
    for row in survey:
        scope = "" if row["level"] == "total" else row["group"]
        survey_data[scope][row["wave"]] = metrics(row) if row["publication_status"] == "available" else None

    raw_team = rows(processed / "team_metrics.csv",
                    ("team_id", "wave", "division", "respondents", "active", "publication_status",
                     "engagement_n", "recommend_signal_n", *fields))
    hidden = protected_teams(raw_team)
    teams = {}
    for row in raw_team:
        team = row["team_id"]
        record = teams.setdefault(team, {"id": team, "division": row["division"], "waves": {}})
        record["waves"][row["wave"]] = (None if team in hidden else
                                         metrics(row) | {"active": int(row["active"]), "respondents": int(row["respondents"])})

    paired = rows(processed / "paired_dynamics.csv",
                  ("level", "group", "cohort", "metric", "status", "earlier", "later", "delta", "lo", "hi", "n"))
    internal_signals, families = intervals_to_signals(paired, waves)
    signals = [r for r in internal_signals if r["team"] not in hidden]
    tracked_pairs = {}
    for r in paired:
        if (r["level"] in {"total", "division"} and r["cohort"] == "pair" and r["earlier"] == waves[0]
                and r["metric"] == "engagement" and r["status"] == "available"):
            scope = "" if r["level"] == "total" else r["group"]
            tracked_pairs.setdefault(scope, {})[r["later"]] = {x: float(r[x]) for x in ("delta", "lo", "hi")}

    annual_rows = rows(processed / "annual_turnover.csv",
                       ("level", "group", "period_start", "period_end", "turnover", "voluntary_turnover"))
    overall = next(row for row in annual_rows if row["level"] == "total")
    annual = {
        "start": overall["period_start"], "end": overall["period_end"],
        **{
            key: {"" if row["level"] == "total" else row["group"]: float(row[column]) if row[column] else None
                  for row in annual_rows if row["level"] in {"total", "division"}}
            for key, column in (("value", "turnover"), ("voluntary", "voluntary_turnover"))
        },
    }
    model = next(row for row in rows(processed / "exit_logistic_models.csv", ("model", "odds_ratio", "lo", "hi"))
                 if row["model"] == "engagement_adjusted")
    comments, differences = comment_snapshot(processed / "comment_analysis" / "analysis", waves, divisions)
    return {
        "waves": [{"id": wave, "date": wave_dates[wave]} for wave in waves],
        "divisions": divisions, "survey": survey_data, "annual": annual, "paired": tracked_pairs,
        "teams": sorted(teams.values(), key=lambda row: row["id"]), "signals": signals,
        "comments": comments, "topic_differences": differences, "market": market_snapshot(processed, annual),
        "outcome_model": [float(model[key]) if model[key] else None for key in ("odds_ratio", "lo", "hi")],
    }, families


def build(processed, output):
    data, families = build_snapshot(processed)
    text = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = (WEB / "template.html").read_text(encoding="utf-8")
    for token, source in (("/*STYLE*/", (WEB / "style.css").read_text(encoding="utf-8")),
                          ("/*SCRIPT*/", (WEB / "app.js").read_text(encoding="utf-8")), ("/*DATA*/", text)):
        if html.count(token) != 1:
            raise ValueError(f"Expected exactly one dashboard marker {token}")
        html = html.replace(token, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_text(output, html)
    return data, families
