import sqlite3
from contextlib import closing

import pandas as pd

from .contract import PAIR, SMALL_GROUP, TABLES


def unique_employees(frame):
    usable = frame["employee_id"].ne("") & ~frame["employee_id"].duplicated(keep=False)
    return frame.loc[usable].set_index("employee_id")


def employee_dates(identifiers, dates):
    return dates.reindex(identifiers).set_axis(identifiers.index)


def check_joins(audit):
    survey = audit.tables["survey_responses"]
    comments = audit.tables["comments"]
    def keep_pairs(frame):
        return frame.loc[frame["employee_id"].ne("") & frame["wave"].ne(""), PAIR]

    query = """
        SELECT COUNT(*) AS joined,
               COUNT(c.employee_id) AS matched,
               COUNT(*) - COUNT(c.employee_id) AS unmatched,
               COUNT(CASE WHEN c.wave <> s.wave THEN 1 END) AS cross_wave
        FROM survey AS s
        LEFT JOIN comments AS c
        ON s.employee_id = c.employee_id AND s.employee_id <> ''
    """
    with closing(sqlite3.connect(":memory:")) as db:
        for name, frame in (("survey", survey), ("comments", comments)):
            db.execute(f"CREATE TABLE {name} (employee_id TEXT, wave TEXT)")
            db.executemany(f"INSERT INTO {name} VALUES (?, ?)", frame[PAIR].itertuples(index=False, name=None))
        db.row_factory = sqlite3.Row
        audit.diagnostics["joins"] = {
            "employee_id": dict(db.execute(query).fetchone()),
            "employee_id + wave": dict(db.execute(
                query + " AND s.wave = c.wave AND s.wave <> ''"
            ).fetchone()),
        }
    survey_pairs = keep_pairs(survey)
    comment_pairs = keep_pairs(comments)
    audit.profiles["survey_responses"]["pairs"] = len(survey_pairs.drop_duplicates())
    audit.profiles["comments"]["pairs"] = len(comment_pairs.drop_duplicates())
    for (employee, wave), row in (
        survey_pairs.value_counts().rename("responses").to_frame()
        .join(comment_pairs.value_counts().rename("comments"), how="inner").sort_index().iterrows()
    ):
        responses, count = int(row["responses"]), int(row["comments"])
        if max(responses, count) > 1:
            audit.log("join_multiplier", "survey_responses + comments", grain="employee_wave",
                      key={"employee_id": employee, "wave": wave}, responses=responses,
                      comments=count, joined_rows=responses * count)
    missing = ~pd.MultiIndex.from_frame(comments[PAIR]).isin(pd.MultiIndex.from_frame(survey_pairs))
    audit.mark("comment_no_response", "comments",
               missing & comments["employee_id"].ne("") & comments["wave"].ne(""), PAIR)


def check_employment(audit):
    employees = audit.tables["employees"]
    roster = unique_employees(employees)
    hire = (
        audit.dates["employees"].loc[employees["employee_id"].isin(roster.index)]
        .set_axis(roster.index)
    )
    exits = audit.tables.get("exits")
    departure = None
    if exits is not None:
        exit_ids = unique_employees(exits).index
        departure = audit.dates["exits"].loc[exits["employee_id"].isin(exit_ids)].set_axis(exit_ids)

    for name in ("survey_responses", "comments", "exits"):
        if name not in audit.tables:
            continue
        frame = audit.tables[name]
        employee = frame["employee_id"]
        comments = name == "comments"
        audit.mark("orphan_comment_employee" if comments else "orphan_employee", name,
                   employee.ne("") & ~employee.isin(employees["employee_id"]), ("employee_id",))
        event = audit.dates[name]
        hired = employee_dates(employee, hire)
        before_hire = event.notna() & hired.notna() & event.lt(hired)
        audit.mark("comment_outside_employment" if comments else "before_hire",
                   name, before_hire, ("employee_id", TABLES[name].date_column), relation="before_hire")
        if name == "exits" or departure is None:
            continue
        departed = employee_dates(employee, departure)
        after_exit = event.notna() & departed.notna() & event.gt(departed)
        same_day = event.notna() & departed.notna() & event.eq(departed)
        audit.mark("comment_outside_employment" if comments else "after_exit",
                   name, after_exit, ("employee_id", "wave", "submitted_at"), relation="after_exit")
        audit.mark("on_exit_day", name, same_day, ("employee_id", "wave", "submitted_at"))

    if "comments" in audit.tables:
        comments = audit.tables["comments"]
        known = comments["employee_id"].isin(roster.index)
        for col in ("team_id", "division"):
            recorded = comments["employee_id"].map(roster[col])
            different = known & comments[col].ne("") & comments[col].ne(recorded)
            for idx in comments.index[different]:
                row = comments.loc[idx]
                audit.log("comment_org_mismatch", "comments", record=int(idx),
                          key={"comment_id": row["comment_id"]}, employee_id=row["employee_id"],
                          wave=row["wave"], field=col, comment_value=row[col], roster_value=recorded[idx])


def check_headcounts(audit):
    if audit.critical("employees", "exits") or not {"employees", "exits"} <= audit.tables.keys():
        return
    employees = audit.tables["employees"]
    hires = audit.dates["employees"]
    departures = audit.dates["exits"].set_axis(audit.tables["exits"]["employee_id"])
    employee_exit = employee_dates(employees["employee_id"], departures)
    teams = employees["team_id"]
    all_teams = sorted(teams.unique())
    audit.diagnostics["small_roster"] = teams.value_counts().loc[lambda n: n < SMALL_GROUP].sort_index().to_dict()
    respondents = audit.tables.get("survey_responses")
    if audit.critical("survey_responses"):
        respondents = None

    waves = {}
    for wave, day in audit.wave_dates.items():
        cutoff = pd.Timestamp(day)
        hired = hires.le(cutoff)
        active = hired & (employee_exit.isna() | employee_exit.gt(cutoff))
        on_exit_day = hired & employee_exit.eq(cutoff)
        row = {"date": day.isoformat(), "active": int(active.sum()), "on_exit_day": int(on_exit_day.sum())}
        by_team = teams.loc[active].value_counts().reindex(all_teams, fill_value=0)
        row["teams"] = {}
        if respondents is not None:
            received_by_cutoff = (
                respondents["wave"].eq(wave) & audit.dates["survey_responses"].le(cutoff)
            )
            has_response = employees["employee_id"].isin(respondents.loc[received_by_cutoff, "employee_id"])
            responded = active & has_response
            by_response = teams.loc[responded].value_counts().reindex(all_teams, fill_value=0)
            row["respondents"] = int(responded.sum())
            row["respondents_on_exit_day"] = int((on_exit_day & has_response).sum())
        for team in all_teams:
            counts = {"active": int(by_team[team])}
            if counts["active"] < SMALL_GROUP:
                audit.log("small_active", "employees", grain="team_wave", key={"team_id": team, "wave": wave},
                          active=counts["active"])
            if respondents is not None:
                counts["respondents"] = int(by_response[team])
                if counts["respondents"] < SMALL_GROUP:
                    audit.log("small_response", "survey_responses", grain="team_wave",
                              key={"team_id": team, "wave": wave}, **counts)
            row["teams"][team] = counts
        waves[wave] = row
    audit.diagnostics["headcounts"] = waves


def check_relationships(audit):
    if "employees" in audit.tables:
        check_employment(audit)
    if {"survey_responses", "comments"} <= audit.tables.keys():
        check_joins(audit)
    check_headcounts(audit)
