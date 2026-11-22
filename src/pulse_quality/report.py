import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from .audit import RULES
from .contract import EXIT_END, EXIT_START, SMALL_GROUP, TABLES, TRAILING_YEAR_START


def table(headers, rows):
    def line(cells):
        return "| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ").replace("\r", "") for cell in cells) + " |"

    return "\n".join([line(headers), line("---" for _ in headers), *(line(row) for row in rows)])


def section(lines, title, *blocks):
    lines.extend([f"## {title}", ""])
    for block in blocks:
        lines.extend([block, ""])


def files_profile(audit):
    rows = []
    for name, profile in audit.profiles.items():
        key = " + ".join(TABLES[name].key)
        unique = ", ".join(f"`{col}` {count}" for col, count in profile["unique"].items())
        blanks = ", ".join(f"`{col}` {count}" for col, count in profile["blank"].items() if count)
        dates = " — ".join(profile.get("dates", []))
        rows.append((
            f"`{name}.csv`", profile["rows"], profile["columns"], f"`{key}` ({profile['key_count']})",
            unique or "—", profile.get("pairs", "—"), blanks or "—", dates or "—",
        ))
    return table(("Файл", "Строки", "Поля", "Ключ и уникальные значения", "Уникальные ID",
                  "Пар сотрудник + волна", "Пропуски", "Диапазон валидных дат"), rows)


def issue_summary(audit):
    groups = defaultdict(list)
    for issue in audit.issues:
        groups[issue["rule"], issue["table"]].append(issue)
    rows = []
    for (rule, source), issues in sorted(groups.items(), key=lambda pair: (
        ("critical", "review", "info").index(RULES[pair[0][0]][0]), pair[0],
    )):
        severity, category, title, impact = RULES[rule]
        records = {item["record"] for item in issues if item["grain"] == "record"}
        records.update(record for item in issues for record in item["evidence"].get("records", []))
        affected = len(records) or sum(item["evidence"].get("affected_records", 0) for item in issues)
        rows.append((f"`{rule}` · {title}", f"`{source}`", f"{severity} / {category}",
                     affected if affected else "—", len(issues), impact))
    return table(("Правило", "Источник", "Уровень / категория", "Записи CSV", "Срабатывания", "Влияние"), rows)


def scores_profile(audit):
    profile = audit.profiles.get("survey_responses")
    if profile is None:
        return "Профиль оценок недоступен."
    rows = (
        (f"`{col}`", f"{item['lower']}–10", item["valid"],
         item["min"] if item["min"] is not None else "—",
         item["max"] if item["max"] is not None else "—",
         profile["blank"][col], item["invalid"])
        for col, item in profile["scores"].items()
    )
    by_wave = [
        (wave, item["size"], item["sum"], f"{item['sum'] / item['size']:.1%}")
        for wave, item in sorted(profile["pay_by_wave"].items())
    ]
    return "\n\n".join([
        table(("Поле", "Шкала", "Валидных", "Мин.", "Макс.", "Пропуски", "Вне шкалы или неверный тип"), rows),
        "Пропуски `q_pay` по волнам. В знаменателе все сырые ответы соответствующей волны.",
        table(("Волна", "Ответы", "Пропуски оплаты", "Доля"), by_wave),
    ])


def repeats_and_joins(audit):
    pair_rules = {"survey_pair_conflict", "survey_pair_duplicate", "comment_pair_multiple"}
    pairs = [item for item in audit.issues if item["rule"] in pair_rules]
    grouped = defaultdict(list)
    for item in pairs:
        grouped[item["table"], item["key"]["wave"]].append(item)
    rows = [
        (source, wave, len(items), sum(len(i["evidence"]["records"]) for i in items),
         sum(i["rule"] == "survey_pair_conflict" for i in items),
         sum("q_recommend" in i["evidence"]["differing_fields"] for i in items))
        for (source, wave), items in sorted(grouped.items())
    ]
    sections = [
        table(("Источник", "Волна", "Повторяющиеся пары", "Строки в парах",
               "Пар с расхождением в оценках", "Расхождение в q_recommend"), rows)
        if rows else "Повторяющиеся пары не выявлены в загруженных таблицах."
    ]
    comment_profile = audit.profiles.get("comments")
    if comment_profile:
        sections.append(
            f"В комментариях {comment_profile['texts']} различных непустых текстов после удаления краевых пробелов. "
            "Одинаковый текст у разных сотрудников может отражать шаблоны синтетического набора. "
            "Текст сам по себе не определяет уникальность комментария."
        )
    joins = audit.diagnostics.get("joins")
    if joins is None:
        return [*sections, "Кардинальность соединений недоступна."]
    base = audit.profiles["survey_responses"]["rows"]
    join_rows = [
        (f"`{key}`", base, info["joined"], info["joined"] - base,
         info["matched"], info["unmatched"], info["cross_wave"])
        for key, info in joins.items()
    ]
    multipliers = [item["evidence"] for item in audit.issues if item["rule"] == "join_multiplier"]
    many_many = sum(i["responses"] > 1 and i["comments"] > 1 for i in multipliers)
    sections.extend([
        table(("Ключ LEFT JOIN", "Ответы до", "Строки после", "Лишних строк", "Соответствий",
               "Ответов без комментария", "Соответствий между волнами"), join_rows),
        f"Число строк на совпавшем ключе равно произведению числа ответов и комментариев. "
        f"Даже по паре сотрудник + волна несколько совпадений возникают у {len(multipliers)} пар. "
        f"Соединение много-ко-многим приходится на {many_many} из них. "
        "Ответы и комментарии стоит хранить отдельными фактами. Для анализа на уровне пары "
        "сначала согласовать выбор ответа и агрегировать комментарии. "
        "Справочник сотрудников присоединять с проверкой `many_to_one`, события увольнений "
        "использовать как отдельный факт с временным ограничением.",
    ])
    return sections


def headcounts(audit):
    waves = audit.diagnostics.get("headcounts")
    if waves is None:
        return ["Численность недоступна. Проверьте критические ошибки и наличие реестра и увольнений."]
    small_roster = audit.diagnostics["small_roster"]
    rows, teams = [], []
    for wave, item in waves.items():
        with_exit_day = item["respondents"] + item["respondents_on_exit_day"] if "respondents" in item else "—"
        rows.append((wave, item["date"], item["active"], item["on_exit_day"],
                     item["active"] + item["on_exit_day"], item.get("respondents", "—"), with_exit_day))
        for team, counts in item["teams"].items():
            respondents = counts.get("respondents")
            if counts["active"] < SMALL_GROUP or (respondents is not None and respondents < SMALL_GROUP):
                teams.append((wave, team, counts["active"], respondents if respondents is not None else "—"))
    roster_counts = ", ".join(f"{key} — {count}" for key, count in small_roster.items()) or "таких команд нет"
    return [
        "Дата среза каждой волны задана по наблюдаемой дате отправки. Базовое правило "
        "`hire_date <= дата среза < exit_date`. Для сотрудника без записи об увольнении применяется "
        "статус из словаря. Число участников учитывает уникальных сотрудников из активного состава с ответом "
        "к дате среза. Вариант включения увольняющихся в тот же день показан отдельно.",
        table(("Волна", "Дата", "Активные", "Ушли в дату среза", "Активные с включением дня ухода",
               "Уникальные активные участники", "Участники с включением дня ухода"), rows),
        f"Команды с численностью менее {SMALL_GROUP} в реестре — {roster_counts}. "
        f"Ниже срезы с менее чем {SMALL_GROUP} активными сотрудниками или участниками. "
        "Один ответ в команде из пяти участников соответствует 20 процентным пунктам бинарной доли.",
        table(("Волна", "Команда", "Активные", "Активные участники"), teams) if teams else "Таких срезов нет.",
        "Исторический состав восстановлен при допущениях о полноте реестра и увольнений, одном трудовом "
        "эпизоде на сотрудника и стабильной принадлежности к команде. Полей истории переводов нет. "
        "Для опубликованных показателей нужны подтверждённые даты волн и порог минимальной выборки.",
    ]


def exit_coverage(audit):
    months = audit.diagnostics.get("exits_by_month")
    if months is None:
        return ["Данные об увольнениях недоступны."]
    days = (EXIT_END - EXIT_START).days + 1
    outside_year = sum(item["rule"] == "exit_outside_12_months" for item in audit.issues)
    return [
        f"Окно словаря `{EXIT_START}` — `{EXIT_END}` включает {days} календарных дней и 13 месяцев. "
        f"При отсечении до 12 календарных месяцев `{TRAILING_YEAR_START}` — `{EXIT_END}` "
        f"{outside_year} событий остаются за пределами годового числителя. "
        "Они нужны для восстановления численности. Первая и последняя наблюдаемая дата события "
        "характеризуют выгрузку, а полнота окна опирается на словарь.",
        table(("Месяц", "Строки с валидной датой"), months.items()),
        "При связывании с ответами использовать только признаки, доступные на дату среза. "
        "Окно наблюдения исхода начинается после ответа и заканчивается не позднее конца выгрузки "
        f"`{EXIT_END}`. Для сравнения волн нужно согласовать общий горизонт и обработку правого цензурирования.",
    ]


def market(audit):
    decisions = audit.diagnostics.get("market")
    if decisions is None:
        return ["Рыночные ориентиры недоступны."]
    rows = (
        (record, row["source"], row["metric"], row["scope"], row["period"],
         f"{row['value']} {row['unit']}", row["methodology"], decision)
        for record, decision in decisions
        for row in [audit.tables["market_context"].loc[record]]
    )
    return [
        "Классификация основана на метаданных выгрузки и предполагаемых внутренних определениях. "
        "Для сравнения нужны совпадение периода, выборки, методики и шкалы. Сегменты рынка "
        "«ИТ-компании» и «розничные сети» не эквивалентны одноимённым функциям внутри банка. "
        "Источник и период измерения сами по себе не подтверждают публикацию до даты среза.",
        table(("Запись", "Источник", "Метрика", "Сегмент", "Период", "Значение", "Методика", "Применимость"), rows),
    ]


def render(audit):
    status = Counter(item["severity"] for item in audit.issues)
    lines = [
        "# Аудит качества данных «Пульс и риск текучести»", "",
        "Проверены синтетические CSV исследования. "
        f"Загружено {len(audit.tables)} из {len(TABLES)} таблиц. "
        f"Срабатывания `critical` — {status['critical']}, `review` — {status['review']}, "
        f"`info` — {status['info']}.",
        "",
        "Расчёты по затронутым данным блокируются до исправления критических ошибок."
        if audit.critical() else
        "Проверки целостности допускают переход к подготовке данных. Перед расчётом метрик по волнам "
        "нужно согласовать выбор ответов при расхождениях и правило для дня увольнения.",
        "",
    ]
    section(lines, "Профиль файлов", files_profile(audit),
            "Ключ для рынка служит для сверки внутри выгрузки. Первичного ключа для рынка словарь не задаёт. "
            "Подсчёт уникальных значений исключает пустые значения.")
    section(lines, "Обнаруженные проблемы и особенности", issue_summary(audit),
            "Столбец «Записи CSV» считает уникальные исходные записи в рамках правила и источника. "
            "Для проверок на уровне таблицы, команды или пары с двух сторон число записей может быть неприменимо. "
            "Записи могут встречаться в нескольких правилах. [Журнал проверок](data_quality_issues.jsonl).")
    zero_rules = (
        ("Повтор первичного ключа", ("duplicate_key",)),
        ("Полное совпадение строк", ("exact_duplicate",)),
        ("Ответ или увольнение до найма, ответ после увольнения", ("before_hire", "after_exit")),
        ("Дата вне месяца волны", ("outside_wave_month", "comment_outside_wave_month")),
        ("Увольнение вне окна словаря", ("exit_out_of_window",)),
        ("Автор ответа или увольнение без реестра сотрудника", ("orphan_employee",)),
        ("Расхождение команды или дивизиона комментария с реестром", ("comment_org_mismatch",)),
    )
    if len(audit.tables) == len(TABLES) and not audit.critical():
        section(lines, "Контрольные проверки", table(("Проверка", "Срабатывания"), (
            (label, sum(item["rule"] in rules for item in audit.issues))
            for label, rules in zero_rules
        )), f"Код внешнего руководителя по словарю обнаружен в {audit.diagnostics['external_managers']} записях реестра.")
    section(lines, "Шкалы и пропуски в ответах", scores_profile(audit))
    section(lines, "Зерно таблиц и размножение строк при объединении", *repeats_and_joins(audit))
    section(lines, "Окно наблюдения увольнений", *exit_coverage(audit))
    section(lines, "Численность на даты волн и малые команды", *headcounts(audit))
    section(lines, "Сопоставимость рыночных ориентиров", *market(audit))
    section(lines, "Правила для следующего этапа",
            "[Правила очистки и открытые решения](../docs/data_quality_rules.md). "
            "[Словарь данных](../docs/Словарь_данных.md). "
            "[Инструкция запуска](../README.md).")
    section(lines, "Контрольные суммы входов", table(("Файл", "SHA-256"), (
        (f"`{name}.csv`", f"`{checksum}`") for name, checksum in audit.sources.items()
    )))
    return "\n".join(lines)


def write_text(path: Path, content: str):
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(content.encode("utf-8"))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_reports(audit, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    journal = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in audit.issues)
    report = render(audit)
    write_text(output / "data_quality_issues.jsonl", journal)
    write_text(output / "data_quality.md", report)
