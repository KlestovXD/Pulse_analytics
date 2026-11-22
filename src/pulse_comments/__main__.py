import argparse
import csv
import json
from pathlib import Path

from .classify import run_classification
from .evaluate import evaluate_review, read_result_rows
from .gemini import GeminiError
from .prepare import build_inventory, export_inventory, select_review_ids, validate_review
from .schema import PROMPTS


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Prepare and classify the unique HR pulse comment texts.")
    parser.add_argument("--comments", type=Path, default=Path("data/processed/comments.csv"))
    parser.add_argument("--raw-comments", type=Path, default=Path("data/raw/comments.csv"))
    parser.add_argument("--employees", type=Path, default=Path("data/raw/employees.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/comment_analysis"))
    parser.add_argument("--sample-ids", type=Path, default=PROMPTS / "review_sample_ids.json")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create the local inventory and blind review template")
    prepare.add_argument("--inventory-only", action="store_true")
    review = commands.add_parser("validate-review", help="Check the independently completed reference file locally")
    review.add_argument("--review", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Compare existing results with the independent reference locally")
    evaluate.add_argument("--review", type=Path, required=True)
    evaluate.add_argument("--results", type=Path, help="Existing comment_labels.jsonl. Defaults to the output directory.")
    analyze = commands.add_parser("analyze", help="Verify complete saved labels and generate local analytical artifacts")
    analyze.add_argument("--review", type=Path, required=True)
    analyze.add_argument("--quantitative-data", type=Path, default=Path("data/processed"))
    analyze.add_argument("--report", type=Path, default=Path("reports/comment_analysis.md"))
    classify = commands.add_parser("classify", help="Request Gemini labels and resume from the local cache")
    classify.add_argument("--review", type=Path, required=True)
    classify.add_argument("--confirm-free-tier", required=True, action="store_true",
                          help="Confirm that this key belongs to a Free Tier project")
    classify.add_argument("--rpm", required=True, type=positive, help="Local request pace within your project's quota")
    classify.add_argument("--max-texts", type=positive, default=62, help="Maximum new texts per invocation")
    args = parser.parse_args()
    try:
        if args.output.resolve().is_relative_to(args.raw_comments.resolve().parent):
            raise ValueError("Use an output directory outside the raw input directory")
        inventory = build_inventory(args.comments, args.raw_comments, args.employees)
        only_inventory = args.command == "prepare" and args.inventory_only
        sample_ids = None if only_inventory else select_review_ids(inventory, args.sample_ids)
        if args.command == "prepare":
            export_inventory(inventory, args.output, sample_ids)
            print(f"{len(inventory.links)} comments, {len(inventory.texts)} distinct texts, {len(sample_ids or [])} for review.")
        else:
            reference = validate_review(args.review, inventory, sample_ids)
            if args.command == "evaluate":
                path = args.results or args.output / "comment_labels.jsonl"
                print(json.dumps(evaluate_review(inventory, reference, read_result_rows(path)), indent=2))
            elif args.command == "analyze":
                from .analysis_pipeline import publish_analysis

                if args.report.suffix.lower() != ".md" or args.report.resolve().is_relative_to(args.raw_comments.resolve().parent):
                    raise ValueError("Choose a Markdown report outside the raw data directory")
                manifest = publish_analysis(inventory, reference, args.review, args.quantitative_data, args.output, args.report)
                print(f"{len(manifest['tables'])} analytical tables in {args.output / 'analysis'}. Report: {args.report}.")
            elif args.command == "classify":
                classified, manual, pending = run_classification(inventory, args.output, args.rpm, args.max_texts)
                print(f"{classified} texts classified (including cached results), {manual} need manual review, {pending} pending.")
            else:
                placeholders = sum(row["status"] == "placeholder" for row in reference.values())
                print(f"Manual review verified: {len(reference) - placeholders} substantive, {placeholders} placeholders.")
    except (OSError, ValueError, csv.Error, GeminiError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
