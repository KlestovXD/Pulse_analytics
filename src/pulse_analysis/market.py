from datetime import date
from pathlib import Path

import pandas as pd


PUBLIC_SOURCES = Path(__file__).resolve().parents[2] / "data/reference/market_public_sources.csv"
COMPARABLE = "сопоставимые"
CONDITIONAL = "условно сопоставимые"
INCOMPARABLE = "несопоставимые"

METHOD_COMPARISON = {
    "текучесть": (
        "Все увольнения делятся на среднюю ежедневную численность. Метод усреднения в ориентире не описан.",
        "Проценты годовой текучести. Сверить способ расчёта знаменателя.",
        "Условный контекст",
    ),
    "текучесть добровольная": (
        "Исключаются вынужденные увольнения. Нужны правила классификации причин и расчёт средней численности.",
        "Проценты добровольной текучести.",
        "Условный контекст",
    ),
    "доля добровольных увольнений": (
        "Доля среди всех увольнений с подтверждённым типом. Нужна сверка определения добровольности.",
        "Проценты. Знаменатель состоит из увольнений.",
        "Условный контекст",
    ),
    "eNPS": (
        "Внутренний пульс с порогами 9–10 и 0–6. Источник описывает ежегодный опрос без указания порогов и выборки.",
        "Пункты eNPS. Сверить формулу и пороги.",
        "Справочно",
    ),
    "индекс вовлечённости": (
        "В пульсе четыре равновесных вопроса. Полный ежегодный опросник отличается составом и может отличаться весами.",
        "Обе шкалы 0–100. Единство шкалы не обеспечивает эквивалентность индексов.",
        "Справочно",
    ),
    "стоимость замещения": (
        "Потребуются расходы на подбор и адаптацию, оценка простоя и годовые зарплаты по ролям.",
        "Доля годовой зарплаты. В выгрузке пульса такой базы нет.",
        "Сценарий",
    ),
}
POPULATIONS = {
    "банковский сектор": "Близкий отраслевой ориентир. Состав банков и выборка неизвестны.",
    "финансовый сектор": "Финансовый сектор шире банковского. Состав респондентов неизвестен.",
    "ИТ-компании": "ИТ-компании и внутренняя ИТ-функция банка имеют разные популяции.",
    "розничные сети": "Розничные сети и розничный дивизион банка имеют разные популяции.",
    "розница": "Розничный сектор и розничный дивизион банка имеют разные популяции.",
    "крупные работодатели": "Межотраслевой сегмент. Неизвестны веса отраслей и групп сотрудников.",
    "все отрасли": "Отраслевой состав шире банковского.",
    "промышленность": "Отрасль отличается от банковского сектора.",
    "специалисты": "Нужно сопоставить определение роли и используемую базу зарплат.",
    "руководители": "Нужно сопоставить тип руководящей роли и используемую базу зарплат.",
}


def compare_market(audit):
    frame = audit.tables["market_context"]
    issues = {}
    for item in audit.issues:
        if item["table"] == "market_context":
            issues.setdefault(item["record"], set()).add(item["rule"])
    rows = []
    for record, row in frame.iterrows():
        metric, scope = row["metric"], row["scope"]
        method, scale, status = METHOD_COMPARISON.get(metric, (
            "Требуется определение метрики.", "Требуется подтверждение шкалы.", "Исключить",
        ))
        if metric in {"текучесть", "текучесть добровольная", "доля добровольных увольнений", "стоимость замещения"}:
            period = f"Ориентир за {row['period']} год. Внутреннее окно увольнений 01.10.2025–30.09.2026."
        else:
            period = f"Ориентир за {row['period']} год. Внутренние волны проведены в 2026 году."
        flags = issues.get(record, set())
        if "benchmark_period" in flags:
            status = "Архив" if row["period"].isdigit() and 0 < int(row["period"]) < 2025 else "Исключить"
        if {"benchmark_method", "benchmark_metric", "benchmark_value", "benchmark_key", "missing_field"} & flags:
            status = "Исключить"
        if "benchmark_method" in flags:
            method = "Методика ориентира не подтверждена по данным источника."
        if status == "Условный контекст" and scope not in {"банковский сектор", "крупные работодатели"}:
            status = "Справочно"
        rows.append({
            "source_record": record, **row.to_dict(), "applicability": status, "method_difference": method,
            "period_difference": period, "population_difference": POPULATIONS.get(scope, "Нужна сверка популяции."),
            "scale_difference": scale, "publication_date_available": False,
            "quality_rules": "|".join(sorted(flags)),
        })
    return pd.DataFrame(rows)


def synthetic_rows(preliminary):
    components = {
        "текучесть": ("Все увольнения", "Среднегодовая численность, способ расчёта не раскрыт"),
        "текучесть добровольная": ("Только добровольные увольнения, классификация не раскрыта",
                                   "Среднегодовая численность, способ расчёта не раскрыт"),
        "доля добровольных увольнений": ("Добровольные увольнения, классификация не раскрыта", "Все увольнения"),
        "eNPS": ("Не раскрыты группы ответов и их пороги", "Не раскрыта база валидных ответов"),
        "индекс вовлечённости": ("Не раскрыты вопросы, баллы и веса", "Не раскрыта база ответов"),
        "стоимость замещения": ("Расходы на подбор, адаптацию и простой, детали оценки неизвестны",
                               "Годовая зарплата по категории роли"),
    }
    for row in preliminary.itertuples(index=False):
        conditional = row.applicability == "Условный контекст"
        numerator, denominator = components.get(row.metric, ("Не раскрыт", "Не раскрыт"))
        if "benchmark_method" in row.quality_rules.split("|"):
            numerator, denominator = "Не раскрыт", "Не раскрыт"
        scenario = conditional and row.metric == "текучесть" and row.scope == "банковский сектор"
        yield {
            "comparison_id": f"synthetic-{row.source_record:02}", "source_id": "raw-market",
            "source": row.source, "source_type": "synthetic", "source_record": row.source_record,
            "metric": row.metric, "scope": row.scope, "period": row.period,
            "period_start": "", "period_end": "", "published": "", "source_url": "",
            "value": pd.to_numeric(row.value, errors="coerce"), "unit": row.unit,
            "sample": "Выборка и география не подтверждены",
            "numerator": numerator, "denominator": denominator, "methodology": row.methodology,
            "status": CONDITIONAL if conditional else INCOMPARABLE,
            "comparison_mode": "scenario" if scenario else "none",
            "limitations": "Публичный первоисточник по реквизитам строки не установлен. "
            + " ".join((row.method_difference, row.period_difference, row.population_difference, row.scale_difference)),
        }


def internal_market_metrics(annual, waves):
    annual = annual.loc[annual["level"].eq("total")].iloc[0]
    wave = waves.loc[waves["level"].eq("total")].sort_values("wave_date").iloc[-1]
    year = tuple(pd.Timestamp(annual[key]).date().isoformat() for key in ("period_start", "period_end"))
    day = pd.Timestamp(wave["wave_date"]).date().isoformat()
    share = 100 * annual["voluntary_exits"] / annual["all_exits"] if annual["all_exits"] else None
    return {
        "текучесть": (annual["turnover"], "%", *year, "annual_turnover.csv", annual["all_exits"], annual["mean_headcount"]),
        "текучесть добровольная": (annual["voluntary_turnover"], "%", *year, "annual_turnover.csv",
                                  annual["voluntary_exits"], annual["mean_headcount"]),
        "доля добровольных увольнений": (share, "%", *year, "annual_turnover.csv",
                                        annual["voluntary_exits"], annual["all_exits"]),
        "eNPS": (wave["recommend_signal"], "пункты eNPS", day, day, "wave_metrics.csv", None, wave["recommend_signal_n"]),
        "индекс вовлечённости": (wave["engagement"], "баллы 0–100", day, day, "wave_metrics.csv", None, wave["engagement_n"]),
    }


def publication_timing(published, cutoff):
    if not published:
        return "unknown"
    frequency = {4: "Y", 7: "M", 10: "D"}[len(published)]
    period = pd.Period(published, freq=frequency)
    if period.end_time.date() <= cutoff:
        return "by_cutoff"
    return "after_cutoff" if period.start_time.date() > cutoff else "unknown"


def build_market_comparison(preliminary, annual, waves):
    public = pd.read_csv(PUBLIC_SOURCES, sep=";", keep_default_na=False, dtype={"published": str})
    internal = internal_market_metrics(annual, waves)
    cutoff = date.fromisoformat(internal["текучесть"][3])
    total_fields = ("internal_value", "internal_unit", "internal_period_start", "internal_period_end",
                    "internal_table", "internal_numerator", "internal_denominator")
    rows = []
    for row in [*synthetic_rows(preliminary), *public.to_dict("records")]:
        mode, status, value = row["comparison_mode"], row["status"], row["value"]
        synthetic = row["source_type"] == "synthetic"
        if (status not in {COMPARABLE, CONDITIONAL, INCOMPARABLE}
                or mode not in {"none", "scenario", "absolute", "relative"}
                or (status == INCOMPARABLE and mode != "none")
                or (synthetic and mode not in {"none", "scenario"})
                or (not synthetic and (mode == "scenario" or not row["source_url"].startswith("https://")
                                       or not row["published"]))):
            raise ValueError(f"Invalid source or comparison policy for {row['comparison_id']}")
        total = internal.get(row["metric"], (None,) * len(total_fields))
        internal_value, internal_unit = total[:2]
        if mode != "none" and row["unit"] != internal_unit:
            raise ValueError(f"Comparison has inconsistent units for {row['comparison_id']}")
        delta = internal_value - value if mode != "none" and pd.notna(internal_value) else None
        rows.append({
            **row, "source_record": row.get("source_record"), "period": row.get("period", row["period_start"][:4]),
            "identification": "synthetic_source_unverified" if synthetic else "public_primary_source",
            "publication_timing": publication_timing(row["published"], cutoff),
            **dict(zip(total_fields, total)), "internal_scope": "Организация (синтетический кейс)" if internal_unit else "",
            "absolute_difference": delta, "difference_unit": "п.п." if delta is not None and internal_unit == "%" else
            (internal_unit if delta is not None else ""),
            "relative_difference_pct": 100 * delta / value if delta is not None and mode == "relative" and value > 0 else None,
        })
    return pd.DataFrame(rows)
