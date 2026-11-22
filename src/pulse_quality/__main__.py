import argparse
from collections import Counter
from pathlib import Path

from .audit import run_audit
from .report import write_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the five raw CSV files and write quality reports.")
    parser.add_argument("--data", type=Path, default=Path("data/raw"), help="Input directory")
    parser.add_argument("--output", type=Path, default=Path("reports"), help="Output directory")
    args = parser.parse_args()
    audit = run_audit(args.data)
    write_reports(audit, args.output)
    counts = Counter(issue["severity"] for issue in audit.issues)
    print(f"{args.output}: critical={counts['critical']} review={counts['review']} info={counts['info']}")
    return int(audit.critical())


if __name__ == "__main__":
    raise SystemExit(main())
