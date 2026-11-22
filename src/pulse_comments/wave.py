"""Run the existing comment classifier on an independently stored monthly wave."""

import argparse
import csv
import json
import os
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import date
from pathlib import Path
from tempfile import mkstemp

from src.pulse_analysis.prepare import prepare_comments
from src.pulse_dashboard.review import TEMPLATE, apply_review_decisions, followup_items, write_followup_template
from src.pulse_quality.audit import run_audit
from src.pulse_quality.report import table, write_text

from .__main__ import positive
from .analysis_pipeline import verify_snapshot
from .analytics import descriptive_tables
from .classify import export_labels, json_lines, run_classification
from .evaluate import evaluate_review, score_categories
from .gemini import MODEL_ID, GeminiClient, GeminiError
from .prepare import build_inventory, export_inventory, select_review_ids, validate_review, write_csv
from .schema import INSTRUCTION_HASH, INSTRUCTION_VERSION, json_loads


PRODUCTION = "google-gemini-developer-api"
MOCK = "mock-gemini-api"
RUN_FILE = "wave-run.json"
AUDIT_FILE = "audit_issues.jsonl"


def store_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare_wave(raw, output, wave_date, snapshot_end=None, *, samples=None, provider=PRODUCTION):
    if output.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Choose a wave output directory outside the raw bundle")
    state_file = output / RUN_FILE
    if output.exists() and not state_file.is_file() and any(path.name != AUDIT_FILE for path in output.iterdir()):
        raise ValueError("This directory already belongs to another workflow. Choose a separate wave output directory.")
    audit = run_audit(raw, wave_date=wave_date, snapshot_end=snapshot_end)
    state = {
        "wave_date": wave_date.isoformat(), "snapshot_end": audit.snapshot_end.isoformat(),
        "input_sha256": audit.sources, "provider": provider, "model": MODEL_ID,
        "instruction_version": INSTRUCTION_VERSION, "instruction_sha256": INSTRUCTION_HASH,
    }
    if state_file.is_file() and json_loads(state_file.read_text(encoding="utf-8")) != state:
        raise ValueError("The input, cutoff, or classification version has changed. Choose a new wave output directory.")
    output.mkdir(parents=True, exist_ok=True)
    write_text(output / AUDIT_FILE, json_lines(audit.issues))
    if audit.critical():
        raise ValueError(f"Wave audit has critical errors. Inspect {output / AUDIT_FILE}.")
    store_json(state_file, state)
    prepared = output / "comments_prepared.csv"
    write_text(prepared, prepare_comments(audit).to_csv(index=False, sep=";", lineterminator="\n", date_format="%Y-%m-%d"))
    inventory = build_inventory(prepared, raw / "comments.csv", raw / "employees.csv")
    sample_ids = select_review_ids(inventory, samples, size=None) if samples else None
    export_inventory(inventory, output, sample_ids)
    return inventory, state


def initialize_cache(cache, seed=None):
    if seed is not None and seed.resolve() == cache.resolve():
        raise ValueError("A seed cache is read-only. Set a separate working cache.")
    if cache.exists() or seed is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(dir=cache.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with closing(sqlite3.connect(seed.resolve().as_uri() + "?mode=ro", uri=True)) as original:
            with closing(sqlite3.connect(temporary)) as clone:
                original.backup(clone)
        # An existing working cache always takes precedence over the seed.
        if not cache.exists():
            temporary.rename(cache)
    finally:
        temporary.unlink(missing_ok=True)


def verified_wave(inventory, state, output):
    snapshots = [output / name for name in ("labels_by_text.jsonl", "comment_labels.jsonl", "manual_review_by_text.jsonl")]
    if not any(path.exists() for path in snapshots):
        export_labels(inventory, output, {}, {})
    elif not all(path.is_file() for path in snapshots):
        raise ValueError("Result snapshot is incomplete. Resume using the same working cache.")
    records, labels, fingerprints = verify_snapshot(inventory, output, allow_partial=True)
    if any(annotation["provider"] != state["provider"] for annotation in labels.values()):
        raise ValueError("Provider mismatch. Use separate caches and outputs for simulations.")
    templates = followup_items(inventory, labels, [], conservative=True)
    return records, labels, fingerprints, templates


def periodic_review(inventory, labels, records, samples, review):
    if samples is None and review is None:
        return None
    if samples is None or review is None:
        raise ValueError("A periodic comparison requires both --sample-ids and --review")
    reference = validate_review(review, inventory, select_review_ids(inventory, samples, size=None))
    evaluation = evaluate_review(inventory, reference, records)
    if evaluation["substantive_agreement"] is not None:
        pairs = [(key, row["label"], labels[key]["label"]) for key, row in reference.items() if row["status"] == "substantive"]
        _, by_class, by_field = score_categories(pairs)
        evaluation.update(class_metrics=by_class, categorical_agreement=by_field)
    return evaluation


def report_wave(inventory, state, output, *, samples=None, review=None):
    records, labels, fingerprints, templates = verified_wave(inventory, state, output)
    counts = Counter(row["label_status"] for row in records)
    text_counts = Counter(row["content_status"] for row in inventory.texts.values())
    classified = len(labels)
    unresolved = {row["text_id"] for row in records if row["label_status"] == "needs_manual_review"}
    pending = text_counts["substantive"] - classified - len(unresolved)
    audit = Counter(json_loads(line)["severity"] for line in (output / AUDIT_FILE).read_text(encoding="utf-8").splitlines())
    evaluation = periodic_review(inventory, labels, records, samples, review)
    volumes, distributions = descriptive_tables(records, labels, allow_partial=True)
    if volumes:
        write_csv(output / "comment_volume.csv", volumes[0], volumes, bom=True)
        write_csv(output / "comment_distribution.csv", distributions[0], distributions, bom=True)
    else:
        write_csv(output / "comment_volume.csv", (
            "level", "wave", "division", "comments", "substantive_comments", "classified_substantive_comments",
            "confirmed_authors", "substantive_confirmed_authors", "status",
        ), [], bom=True)
        write_csv(output / "comment_distribution.csv", (
            "level", "wave", "division", "dimension", "category", "comments", "unique_confirmed_authors",
            "substantive_denominator", "share", "status",
        ), [], bom=True)
    write_followup_template(output / TEMPLATE, templates)
    coverage = {
        "records": len(records), "unique_texts": len(inventory.texts), "substantive_texts": text_counts["substantive"],
        "classified_texts": classified, "pending_texts": pending, "technical_review_texts": len(unresolved),
        "semantic_review_texts": len(templates), "rows_by_status": dict(counts),
        "unconfirmed_author_rows": sum(row["author_status"] != "confirmed" for row in records),
    }
    summary = {
        **state, "coverage": coverage, "audit": dict(audit), "result_sha256": fingerprints,
        "reported_model_versions": sorted({row["model_version"] for row in labels.values()}),
        "periodic_review": evaluation,
    }
    store_json(output / "wave-summary.json", summary)
    origin = "**Синтетическая имитация API. Метки созданы тестовым адаптером.**" if state["provider"] == MOCK else (
        f"Модель для запуска `{state['model']}`, версия инструкции `{state['instruction_version']}`."
    )
    headline = table(("Показатель", "Результат"), (
        ("Исходные записи / уникальные тексты", f"{len(records)} / {len(inventory.texts)}"),
        ("Содержательные уникальные / классифицированы", f"{text_counts['substantive']} / {classified}"),
        ("Ожидают / техническая очередь / смысловая очередь", f"{pending} / {len(unresolved)} / {len(templates)}"),
        ("Заглушки / пустые записи", f"{counts['placeholder']} / {counts['empty']}"),
        ("Записи без подтверждённой связи с автором", coverage["unconfirmed_author_rows"]),
        ("Критические / прочие замечания аудита", f"{audit['critical']} / {audit['review'] + audit['info']}"),
    ))
    quality = "Периодическая контрольная сверка не запускалась. Число валидных ответов характеризует покрытие."
    if evaluation is not None:
        agreement = evaluation["substantive_agreement"]
        quality = ("Согласованность по тональности — "
                   f"{agreement['sentiment_accuracy']:.1%}, микро-F1 по темам — {agreement['topics_micro_f1']:.1%} "
                   f"на {agreement['texts']} текстах. Заглушки проверены отдельно."
                   if agreement else "Для контрольной выборки пока отсутствует часть валидных меток. Метрики качества отложены.")
    report = [
        f"# Комментарии волны {state['wave_date']}", "", origin, "", headline, "", quality, "",
        "Тематические доли используют только валидно размеченные содержательные записи. "
        "Покрытие по срезу указано в таблице объёма. Групповые показатели с числом подтверждённых авторов менее 10 скрыты. "
        "Связь автора, волны и подразделения проверена перед группировкой.", "",
        "[Аудит](audit_issues.jsonl) · [Покрытие](comment_volume.csv) · [Темы и тональность](comment_distribution.csv) · "
        "[Техническая очередь](manual_review_by_text.jsonl) · [Смысловая проверка](manual_followup.csv) · "
        "[Параметры и контроль качества](wave-summary.json)", "",
        "Серьёзность, сигналы ухода и неоднозначные случаи требуют ручной проверки перед использованием в решениях. "
        "Доля отрицательных записей описывает комментарии. Следующая контрольная выборка нужна при изменении "
        "модели, инструкции или состава текстов.", "",
    ]
    write_text(output / "wave-summary.md", "\n".join(report))
    return summary


def classify_wave(inventory, state, output, cache, rpm, limit=None, *, seed=None, client_factory=GeminiClient):
    initialize_cache(cache, seed)
    try:
        run_classification(
            inventory, output, rpm, limit or max(1, len(inventory.texts)), client_factory=client_factory,
            cache=cache, provider=state["provider"],
        )
    except GeminiError:
        report_wave(inventory, state, output)
        raise
    return report_wave(inventory, state, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wave-date", type=date.fromisoformat, required=True)
    parser.add_argument("--snapshot-end", type=date.fromisoformat)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--sample-ids", type=Path)
    classify = commands.add_parser("classify")
    classify.add_argument("--cache", type=Path, required=True)
    classify.add_argument("--seed-cache", type=Path)
    classify.add_argument("--rpm", type=positive, required=True)
    classify.add_argument("--max-texts", type=positive)
    classify.add_argument("--confirm-free-tier", required=True, action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--sample-ids", type=Path)
    report.add_argument("--review", type=Path)
    apply = commands.add_parser("apply-review")
    apply.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    try:
        inventory, state = prepare_wave(
            args.raw, args.output, args.wave_date, args.snapshot_end,
            samples=args.sample_ids if args.command == "prepare" else None,
        )
        if args.command == "prepare":
            print(f"Audited {len(inventory.links)} comments and {len(inventory.texts)} distinct texts.")
            return
        if args.command == "classify":
            archived = Path("data/processed/comment_analysis/gemini_cache.sqlite3").resolve()
            if (args.cache.resolve() == archived or args.cache.resolve().is_relative_to(args.raw.resolve())
                    or args.cache.suffix.lower() != ".sqlite3"):
                raise ValueError("Use a separate .sqlite3 working cache outside the raw or archival data")
            summary = classify_wave(inventory, state, args.output, args.cache, args.rpm,
                                    args.max_texts, seed=args.seed_cache)
        elif args.command == "report":
            summary = report_wave(inventory, state, args.output, samples=args.sample_ids, review=args.review)
        else:
            _, labels, fingerprints, items = verified_wave(inventory, state, args.output)
            path, applied = apply_review_decisions(args.output, labels, items, fingerprints["labels_by_text.jsonl"], args.review)
            print(f"{path}: {applied} human decisions.")
            return
        print(json.dumps(summary["coverage"], ensure_ascii=True))
    except (OSError, ValueError, csv.Error, GeminiError, sqlite3.Error) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
