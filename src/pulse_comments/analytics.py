from collections import Counter, defaultdict
from statistics import mean

from src.pulse_quality.contract import SMALL_GROUP

from .evaluate import ratio
from .schema import SCHEMA


TOPIC_TO_SCORE = {
    "оплата": "q_pay", "признание": "q_recognition", "развитие": "q_growth", "руководитель": "q_manager",
    "нагрузка": "q_balance", "условия": "q_conditions", "цели": "q_goals", "инструменты": "q_tools",
}
PROBLEM_SENTIMENTS = {"отрицательная", "смешанная"}
POLARITY_TOPICS = {
    "положительная": "topic_in_positive_comment",
    "отрицательная": "topic_in_negative_comment",
}
NEITHER_POLARITY = {"нейтральная", "неопределенная"}
SERIOUS = {"высокая", "критическая"}


def author_count(rows):
    return len({row["employee_id"] for row in rows if row["author_status"] == "confirmed"})


def dimensions(label):
    yield "topic", label["topics"]
    yield "sentiment", [label["sentiment"]]
    yield "severity", [label["severity"]]
    if label["sentiment"] in PROBLEM_SENTIMENTS:
        yield "topic_in_problem_comment", label["topics"]
    if label["sentiment"] in POLARITY_TOPICS:
        yield POLARITY_TOPICS[label["sentiment"]], label["topics"]


def descriptive_tables(records, labels, *, allow_partial=False):
    topics = SCHEMA["properties"]["topics"]["items"]["enum"]
    categories = {
        "topic": topics, "topic_in_problem_comment": topics,
        **dict.fromkeys(POLARITY_TOPICS.values(), topics),
        "sentiment": SCHEMA["properties"]["sentiment"]["enum"], "severity": SCHEMA["properties"]["severity"]["enum"],
    }
    groups = defaultdict(list)
    for row in records:
        groups["all", "", ""].append(row)
        groups["wave", row["wave"], ""].append(row)
        if row["usable_team_text"]:
            groups["division", "", row["division"]].append(row)
            groups["division_wave", row["wave"], row["division"]].append(row)
    volumes, distributions = [], []
    for (level, wave, division), group in sorted(groups.items()):
        identity = {"level": level, "wave": wave, "division": division}
        original_content = [row for row in group if row["content_status"] == "substantive"]
        content = [row for row in original_content if row["text_id"] in labels] if allow_partial else original_content
        confirmed_authors = author_count(content)
        usable = confirmed_authors >= SMALL_GROUP
        volume = identity | {
            "comments": len(group) if usable else None,
            "substantive_comments": len(original_content) if usable else None,
            "confirmed_authors": author_count(group) if usable else None,
            "substantive_confirmed_authors": confirmed_authors if usable else None,
            "status": "available" if usable else "suppressed_group",
        }
        if allow_partial:
            volume["classified_substantive_comments"] = len(content) if usable else None
        volumes.append(volume)
        counts, authors = Counter(), defaultdict(set)
        by_tone, authors_by_tone = Counter(), defaultdict(set)
        for row in content:
            label = labels[row["text_id"]]["label"]
            for dimension, values in dimensions(label):
                for value in values:
                    key = dimension, value
                    counts[key] += 1
                    if row["author_status"] == "confirmed":
                        authors[key].add(row["employee_id"])
            for topic in label["topics"]:
                key = topic, label["sentiment"]
                by_tone[key] += 1
                if row["author_status"] == "confirmed":
                    authors_by_tone[key].add(row["employee_id"])

        def hidden_remainder(topic, tones):
            return any(by_tone[topic, tone] for tone in tones) and len(
                set().union(*(authors_by_tone[topic, tone] for tone in tones))
            ) < SMALL_GROUP

        for dimension, values in categories.items():
            polarity = next((tone for tone, kind in POLARITY_TOPICS.items() if kind == dimension), None)
            if polarity is not None and level not in {"wave", "division_wave"}:
                continue
            hidden = {value for value in values if counts[dimension, value] and len(authors[dimension, value]) < SMALL_GROUP}
            # An additive total would reveal a single withheld count in an exclusive distribution.
            secondary = set()
            if dimension in {"sentiment", "severity"} and len(hidden) == 1:
                eligible = [value for value in values if counts[dimension, value] and value not in hidden]
                if eligible:
                    secondary.add(min(eligible, key=lambda value: (counts[dimension, value], value)))
            if polarity is not None:
                remainder = NEITHER_POLARITY if polarity == "положительная" else {"смешанная"}
                for topic in values:
                    if (hidden_remainder(topic, remainder)
                            or hidden_remainder(topic, set(categories["sentiment"]) - {polarity})):
                        secondary.add(topic)
            for value in values:
                key = dimension, value
                status = ("suppressed_group" if not usable else
                          "suppressed_cell" if value in hidden else
                          "suppressed_complement" if value in secondary else "available")
                shown = status == "available"
                distributions.append(identity | {
                    "dimension": dimension, "category": value, "comments": counts[key] if shown else None,
                    "unique_confirmed_authors": len(authors[key]) if shown else None,
                    "substantive_denominator": len(content) if usable else None,
                    "share": ratio(counts[key], len(content)) if shown else None, "status": status,
                })
    by_wave_topic = defaultdict(list)
    for row in distributions:
        if row["dimension"] in POLARITY_TOPICS.values():
            by_wave_topic[row["wave"], row["dimension"], row["category"]].append(row)
    for group in by_wave_topic.values():
        overall = next((row for row in group if row["level"] == "wave"), None)
        divisions = [row for row in group if row["level"] == "division_wave"]
        if overall is None or overall["status"] != "available" or sum(
            row["status"] != "available" for row in divisions
        ) != 1:
            continue
        visible = [row for row in divisions if row["status"] == "available"]
        complement = min(visible, key=lambda row: (row["comments"] == 0, row["comments"]), default=overall)
        complement.update(status="suppressed_complement", comments=None, unique_confirmed_authors=None, share=None)
    return volumes, distributions


def keyed(rows, columns):
    indexed = {}
    for row in rows:
        key = tuple(row[col] for col in columns)
        if not all(key) or key in indexed:
            raise ValueError(f"Missing or repeated key in {columns}")
        indexed[key] = row
    return indexed


def scored_comments(records, source_rows, workforce, scores):
    source = keyed(source_rows, ("comment_id",))
    people = keyed(workforce, ("employee_id", "wave"))
    survey = keyed(scores, ("employee_id", "wave"))
    linked, active_people = [], {}
    for row in records:
        if not row["usable_team_text"]:
            continue
        pair = row["employee_id"], row["wave"]
        employee = people.get(pair)
        comment_date = source[(row["comment_id"],)]["submitted_at"]
        if employee is None or employee["division"] != row["division"] or employee["team_id"] != row["team_id"]:
            raise ValueError("Comment organization disagrees with the audited workforce table")
        if employee["responded"] != "True" or not comment_date or comment_date > employee["wave_date"]:
            continue
        if pair not in survey:
            raise ValueError("An active commenter has no survey row at the wave cutoff")
        linked.append(row | {"submitted_at": comment_date})
        active_people[pair] = survey[pair]
    return linked, active_people, people


def topic_score_table(linked, people, labels):
    mentions, adverse_mentions = defaultdict(set), defaultdict(set)
    for row in linked:
        pair = row["employee_id"], row["wave"]
        label = labels[row["text_id"]]["label"]
        mentions[pair].update(label["topics"])
        if label["sentiment"] in PROBLEM_SENTIMENTS:
            adverse_mentions[pair].update(label["topics"])
    waves = ("", *sorted({wave for _, wave in people}))
    rows = []
    for wave in waves:
        eligible = [pair for pair in people if not wave or pair[1] == wave]
        for topic, question in TOPIC_TO_SCORE.items():
            for mode, by_pair in (("any_mention", mentions), ("problem_comment", adverse_mentions)):
                measures = [[], []]
                for pair in eligible:
                    value = people[pair][question]
                    # In the adverse comparison, a positive-only mention is omitted from the baseline.
                    if value and (mode != "problem_comment" or topic not in mentions[pair] or topic in by_pair[pair]):
                        measures[int(topic in by_pair[pair])].append((pair[0], float(value)))
                baseline, exposed = measures
                enough = all(len({employee for employee, _ in group}) >= SMALL_GROUP for group in measures)
                reference = mean(value for _, value in baseline) if enough else None
                mentioned = mean(value for _, value in exposed) if enough else None
                rows.append({
                    "wave": wave, "topic": topic, "question": question, "comparison": mode,
                    "mentions_employee_waves": len(exposed) if enough else None,
                    "other_employee_waves": len(baseline) if enough else None,
                    "mentions_unique_authors": len({employee for employee, _ in exposed}) if enough else None,
                    "other_unique_authors": len({employee for employee, _ in baseline}) if enough else None,
                    "mentions_mean": mentioned, "other_mean": reference,
                    "difference": mentioned - reference if enough else None,
                    "status": "available" if enough else "suppressed_group",
                })
    return rows


def turnover_table(linked, workforce, followups, annual, labels):
    cohorts = keyed(followups, ("employee_id", "wave"))
    events, comments90, comments_annual = defaultdict(list), defaultdict(list), defaultdict(list)
    for pair, outcome in cohorts.items():
        employee = workforce.get(pair)
        if employee is None or not outcome["as_of"] or outcome["exit_within_90d"] not in {"True", "False"}:
            raise ValueError("Invalid follow-up linkage")
        events[pair[1], employee["division"]].append(outcome["exit_within_90d"] == "True")
    for row in linked:
        pair = row["employee_id"], row["wave"]
        cohort = cohorts.get(pair)
        if cohort is not None and row["submitted_at"] <= cohort["as_of"]:
            comments90[row["wave"], row["division"]].append(row)
        comments_annual[row["division"]].append(row)
    results = []
    for (wave, division), outcomes in sorted(events.items()):
        comments = comments90[wave, division]
        enough_cohort = len(outcomes) >= SMALL_GROUP
        metrics = group_comment_metrics(comments, labels, enough_cohort)
        results.append({"measure": "exit_90d", "wave": wave, "division": division,
                        "period_start": "", "period_end": "", "exits": sum(outcomes) if enough_cohort else None,
                        "turnover_denominator": len(outcomes) if enough_cohort else None,
                        "exit_rate": ratio(sum(outcomes), len(outcomes)) if enough_cohort else None,
                        **metrics})
    for row in annual:
        if row["level"] != "division":
            continue
        division = row["group"]
        comments = [r for r in comments_annual[division] if row["period_start"] <= r["submitted_at"] <= row["period_end"]]
        size = float(row["mean_headcount"]) if row["mean_headcount"] else 0
        enough_cohort = size >= SMALL_GROUP
        metrics = group_comment_metrics(comments, labels, enough_cohort)
        results.append({"measure": "annual_turnover", "wave": "", "division": division,
                        "period_start": row["period_start"], "period_end": row["period_end"],
                        "exits": int(row["all_exits"]) if enough_cohort and row["all_exits"] else None,
                        "turnover_denominator": size if enough_cohort else None,
                        "exit_rate": float(row["turnover"]) / 100 if enough_cohort and row["turnover"] else None, **metrics})
    return results


def group_comment_metrics(rows, labels, enough_cohort):
    authors = author_count(rows)
    enough = enough_cohort and authors >= SMALL_GROUP
    negative = [row for row in rows if labels[row["text_id"]]["label"]["sentiment"] == "отрицательная"]
    serious = [row for row in rows if labels[row["text_id"]]["label"]["severity"] in SERIOUS]
    can_show_negative = enough and (not negative or author_count(negative) >= SMALL_GROUP)
    can_show_serious = enough and (not serious or author_count(serious) >= SMALL_GROUP)
    return {
        "substantive_comments": len(rows) if enough else None,
        "unique_comment_authors": authors if enough else None,
        "negative_comments": len(negative) if can_show_negative else None,
        "negative_share": ratio(len(negative), len(rows)) if can_show_negative else None,
        "high_or_critical_comments": len(serious) if can_show_serious else None,
        "high_or_critical_share": ratio(len(serious), len(rows)) if can_show_serious else None,
        "comment_group_status": "available" if enough else "suppressed_group",
    }
