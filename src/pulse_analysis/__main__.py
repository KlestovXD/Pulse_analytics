import argparse
import hashlib
import json
from pathlib import Path

from src.pulse_quality.contract import PAIR, WAVE_DATES
from src.pulse_quality.report import write_text

from .market import PUBLIC_SOURCES, build_market_comparison, compare_market
from .metrics import (
    annual_turnover, decompose_engagement, paired_metrics, sample_composition, team_metrics,
    wave_metrics, wave_people,
)
from .outcomes import analyze_outcomes
from .prepare import check_key, prepare
from .report import render


def calculate(raw):
    audit, lifecycle, datasets = prepare(raw)
    people = wave_people(datasets["employee_wave"][0], datasets["survey_at_wave"][0])
    waves = wave_metrics(people)
    pairs = paired_metrics(people)
    annual = annual_turnover(lifecycle)
    teams = team_metrics(waves, pairs, annual)
    risks, score_diffs, logistic = analyze_outcomes(datasets)
    public_waves = waves.loc[waves["level"].ne("team_id")].reset_index(drop=True)
    public_annual = annual.loc[annual["level"].ne("team_id")].reset_index(drop=True)
    preliminary_market = compare_market(audit)
    last_wave = list(WAVE_DATES)[-1]
    suppressed = public_waves.loc[public_waves["wave"].eq(last_wave)
                                  & public_waves["publication_status"].ne("available"), ["level", "group"]]
    if len(suppressed):
        hidden = set(suppressed.itertuples(index=False, name=None))
        column_mask = public_annual.columns.difference(["level", "group", "period_start", "period_end"])
        public_annual.loc[
            [key in hidden for key in public_annual[["level", "group"]].itertuples(index=False, name=None)],
            column_mask,
        ] = None
    outputs = {
        "wave_metrics": (public_waves, ("level", "group", "wave")),
        "team_metrics": (teams, ("team_id", "wave")),
        "annual_turnover": (public_annual, ("level", "group")),
        "paired_dynamics": (pairs, ("earlier", "later", "cohort", "level", "group", "metric")),
        "participant_composition": (sample_composition(people), ("wave", "dimension", "group")),
        "engagement_decomposition": (decompose_engagement(people, pairs), ("earlier", "later")),
        "exit_risk_groups": (risks, ("dimension", "group")),
        "exit_score_differences": (score_diffs, ("metric",)),
        "exit_logistic_models": (logistic, ("model",)),
        "market_comparability": (preliminary_market, ("source_record",)),
        "market_comparison": (build_market_comparison(preliminary_market, public_annual, public_waves),
                              ("comparison_id",)),
    }
    for frame, key in outputs.values():
        check_key(frame, key)
    if len(teams) != lifecycle["team_id"].nunique() * len(WAVE_DATES):
        raise ValueError("Team-wave grid does not reconcile")
    hidden_teams = teams.loc[teams["publication_status"].ne("available")].select_dtypes("number")
    if hidden_teams.notna().any().any():
        raise ValueError("Numerical detail found for a suppressed team")
    return datasets | outputs, audit.sources


def publish(raw, processed, report):
    raw, processed, report = raw.resolve(), processed.resolve(), report.resolve()
    if processed.is_relative_to(raw) or report.is_relative_to(raw) or report.suffix.lower() != ".md":
        raise ValueError("Use a separate processed directory and a Markdown report outside the raw directory")
    datasets, sources = calculate(raw)
    markdown = render(datasets, processed, report)
    metadata = {
        "raw_sha256": {f"{name}.csv": digest for name, digest in sources.items()},
        "reference_sha256": {PUBLIC_SOURCES.name: hashlib.sha256(PUBLIC_SOURCES.read_bytes()).hexdigest()},
        "encoding": "UTF-8",
        "delimiter": ";",
        "tables": {},
    }
    processed.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    for name, (frame, key) in datasets.items():
        filename = f"{name}.csv"
        metadata["tables"][filename] = {"rows": len(frame), "key": list(key), "columns": list(frame.columns)}
        write_text(processed / filename, frame.to_csv(
            index=False, sep=";", lineterminator="\n", float_format="%.10g", date_format="%Y-%m-%d",
        ))
    write_text(processed / "manifest.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    write_text(report, markdown)
    return len(datasets)


def main():
    parser = argparse.ArgumentParser(description="Prepare the HR pulse data and generate the quantitative analysis.")
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--report", type=Path, default=Path("reports/quantitative_analysis.md"))
    args = parser.parse_args()
    try:
        count = publish(args.data, args.processed, args.report)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(f"{count} tables in {args.processed}. Report at {args.report}.")


if __name__ == "__main__":
    main()
