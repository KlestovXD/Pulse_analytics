import argparse
from pathlib import Path

from .build import build
from .review import apply_followup, prepare_followup


def main():
    parser = argparse.ArgumentParser(description="Build an offline HR dashboard and manage local review decisions.")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    commands = parser.add_subparsers(dest="command", required=True)
    build_cmd = commands.add_parser("build")
    build_cmd.add_argument("--output", type=Path, default=Path("dashboard/index.html"))
    commands.add_parser("prepare-review")
    apply_cmd = commands.add_parser("apply-review")
    apply_cmd.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            if args.output.suffix.lower() != ".html" or args.output.resolve().is_relative_to(args.raw.resolve()):
                raise ValueError("Choose an HTML output outside the raw directory")
            data, families = build(args.processed, args.output)
            print(f"Dashboard: {args.output}. BH families: {families}. Visible signals: {len(data['signals'])}.")
        else:
            if args.processed.resolve().is_relative_to(args.raw.resolve()):
                raise ValueError("Use a processed directory outside the raw directory")
            path, total = (prepare_followup(args.raw, args.processed) if args.command == "prepare-review"
                           else apply_followup(args.raw, args.processed, args.review))
            print(f"{path}: {total} texts.")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
