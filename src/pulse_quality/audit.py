import csv
import hashlib
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from .contract import (
    BENCHMARKS, ENUMS, EXIT_END, EXIT_START, ID_FORMATS, PAIR,
    SCORES, TABLES, TRAILING_YEAR_START, WAVE_DATES,
)


# Уровень определяет код завершения, категория описывает причину проблемы.
RULES = {
    "input_unreadable": ("critical", "technical", "Чтение или схема CSV", "Таблицу нельзя безопасно загрузить."),
    "empty_table": ("critical", "technical", "Таблица без записей", "Нужно подтвердить полноту выгрузки и смысл нулевого объёма."),
    "trimmed_field": ("info", "technical", "Краевые пробелы в структурированных полях", "Сверять ключи после удаления краевых пробелов."),
    "exact_duplicate": ("review", "technical", "Полностью одинаковые строки", "При очистке сохранить один экземпляр и связь с исходными записями."),
    "missing_key": ("critical", "technical", "Пустой первичный ключ", "Нарушена адресуемость записей."),
    "duplicate_key": ("critical", "technical", "Повтор первичного ключа", "Объединение размножит строки или неоднозначно выберет запись."),
    "benchmark_key": ("review", "methodological", "Повтор ключа сверки ориентира", "Проверить различия в методиках и версиях перед выбором значения."),
    "missing_field": ("review", "technical", "Пустое поле", "Объём пригодных данных зависит от поля и расчёта."),
    "missing_core_field": ("critical", "technical", "Пустое поле для основной связи или сегмента", "Нельзя надёжно связать событие или определить принадлежность."),
    "invalid_id": ("critical", "technical", "Некорректный идентификатор", "Связь или принадлежность сотрудника требуют исправления."),
    "invalid_optional_id": ("review", "technical", "Некорректная ссылка в дополнительных данных", "Не использовать для группировки до сверки."),
    "invalid_enum": ("review", "technical", "Значение вне справочника", "Проверить кодировку категории перед сегментацией."),
    "invalid_core_enum": ("critical", "technical", "Категория ключевого поля вне справочника", "Принадлежность к волне, дивизиону или типу увольнения неоднозначна."),
    "invalid_date": ("critical", "technical", "Ошибочная календарная дата", "Хронологию или численность нельзя восстановить надёжно."),
    "invalid_comment_date": ("review", "technical", "Ошибочная дата комментария", "Отложить привязку комментария до уточнения даты."),
    "alternate_comment_date": ("info", "technical", "Дата комментария в формате ДД.ММ.ГГГГ", "Однозначно нормализуется. Хранить исходное значение."),
    "missing_manager": ("review", "technical", "Менеджер не указан", "Разрезы по руководителям неполны."),
    "invalid_manager_reference": ("review", "technical", "Неподтверждённая ссылка на руководителя", "Проверить внешний код, ID и ссылки на себя."),
    "team_multiple_divisions": ("critical", "technical", "Команда относится к нескольким дивизионам", "Агрегация команд до дивизионов неоднозначна."),
    "hire_after_snapshot": ("critical", "technical", "Найм после даты состояния реестра", "Проверить, к какому срезу относится запись."),
    "missing_pay": ("info", "observed", "Пропуск оценки оплаты", "Меняется знаменатель оценки оплаты и состав выборки индекса."),
    "invalid_score": ("review", "technical", "Оценка вне целочисленной шкалы", "При очистке пометить значение пропуском с причиной. Проверить влияние на индекс и eNPS."),
    "survey_pair_duplicate": ("review", "technical", "Несколько ответов сотрудника в волне без расхождений в оценках", "Нужен один ответ на пару с сохранением происхождения."),
    "survey_pair_conflict": ("review", "technical", "Несколько ответов сотрудника в волне с расхождениями", "Выбор записи может изменить оценки. Порядок ID не доказывает очередность версий."),
    "comment_pair_multiple": ("info", "observed", "Несколько комментариев сотрудника в волне", "Сохранить самостоятельные тексты. Для связи с ответами свести до пары."),
    "comment_placeholder": ("info", "observed", "Комментарий-заглушка", "Учитывать отдельно при расчёте доли содержательных текстов."),
    "orphan_employee": ("critical", "technical", "Сотрудник события отсутствует в реестре", "Нельзя восстановить принадлежность или трудовой статус."),
    "orphan_comment_employee": ("review", "technical", "Автор комментария отсутствует в реестре", "Комментарий не подходит для связи с сотрудником и командой."),
    "comment_no_response": ("review", "technical", "Для комментария отсутствует ответ той же волны", "Нужно проверить отдельную возможность оставить комментарий и условия связки."),
    "comment_org_mismatch": ("review", "technical", "Команда или дивизион расходятся с реестром", "Проверить историю переводов и источник атрибуции."),
    "outside_wave_month": ("critical", "technical", "Ответ датирован вне месяца волны", "Когорту волны и дату признаков нужно уточнить."),
    "comment_outside_wave_month": ("review", "technical", "Комментарий датирован вне месяца волны", "Отложить привязку к волне до проверки."),
    "outside_wave_anchor": ("review", "methodological", "Дата отправки отличается от даты среза волны", "Уточнить окно сбора и дату знаменателя."),
    "after_wave_cutoff": ("critical", "technical", "Запись датирована после среза новой волны", "Проверить дату среза или разделить выгрузки."),
    "before_hire": ("critical", "technical", "Ответ или увольнение раньше найма", "Текущий реестр не подтверждает трудовой эпизод."),
    "after_exit": ("critical", "technical", "Ответ после увольнения", "Проверить трудовой эпизод и исключить признаки после исхода."),
    "comment_outside_employment": ("review", "technical", "Комментарий до найма или после увольнения", "Проверить трудовой эпизод перед связкой."),
    "on_exit_day": ("review", "methodological", "Пульс-ответ или комментарий в день увольнения", "Время событий неизвестно. Проверить чувствительность к правилу дня увольнения."),
    "exit_out_of_window": ("critical", "technical", "Увольнение вне указанного окна", "Событие может изменить знаменатель и числитель."),
    "exit_outside_12_months": ("info", "methodological", "Увольнение в дополнительном месяце выгрузки", "Использовать в истории численности. Ограничить числитель при расчёте за 12 месяцев."),
    "exit_window_length": ("review", "methodological", "Окно в словаре длиннее 12 месяцев", "Согласовать датировку годовой текучести и способ расчёта средней численности."),
    "benchmark_metric": ("review", "technical", "Метрика ориентира вне словаря", "Подтвердить определение метрики."),
    "benchmark_value": ("review", "technical", "Некорректное значение или единица ориентира", "Сравнение отложить до сверки шкалы."),
    "benchmark_period": ("review", "methodological", "Неизвестный, архивный или более поздний период ориентира", "Проверить доступность на дату анализа. Архивные значения оставить для справки."),
    "benchmark_method": ("review", "methodological", "Методика ориентира требует сверки", "Без сопоставимой формулы сравнение количественных уровней недопустимо."),
    "benchmark_publication": ("info", "methodological", "Дата публикации отсутствует в схеме ориентиров", "Период измерения не подтверждает доступность источника на дату волны."),
    "join_multiplier": ("review", "technical", "Несколько совпадений при соединении по сотруднику и волне", "Соединение сырых фактов повторяет ответы и/или комментарии."),
    "small_active": ("info", "observed", "Малочисленная активная команда", "При малом знаменателе доля чувствительна к одному сотруднику."),
    "small_response": ("info", "observed", "Мало активных участников команды", "Оценки и доли по команде нестабильны. Согласовать порог публикации."),
}


@dataclass
class Audit:
    sources: dict = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    dates: dict[str, pd.Series] = field(default_factory=dict)
    issues: list[dict] = field(default_factory=list)
    profiles: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    wave_dates: dict[str, date] = field(default_factory=lambda: WAVE_DATES.copy())
    snapshot_end: date = EXIT_END
    annual_window: bool = True

    def log(self, rule, table, *, grain="record", record=None, key=None, **evidence):
        severity, category, _, _ = RULES[rule]
        self.issues.append({
            "rule": rule, "severity": severity, "category": category,
            "table": table, "grain": grain, "record": record, "key": key, "evidence": evidence,
        })

    def mark(self, rule, table, mask, columns=(), **evidence):
        frame = self.tables[table]
        for record in frame.index[mask]:
            values = frame.loc[record]
            key = {column: values[column] for column in TABLES[table].key}
            details = {column: values[column] for column in columns}
            self.log(rule, table, record=int(record), key=key, **details, **evidence)

    def critical(self, *tables):
        return any(
            item["severity"] == "critical" and (not tables or item["table"] in tables)
            for item in self.issues
        )


class InputError(ValueError):
    def __init__(self, message, record=None):
        super().__init__(message)
        self.record = record


def read_csv(raw, expected):
    rows = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), delimiter=";", strict=True)
    header = next(rows, [])
    if len(header) != len(expected) or set(header) != set(expected):
        raise InputError(f"Ожидались поля {', '.join(expected)}. Получены {', '.join(header)}")
    records = []
    try:
        for number, row in enumerate(rows, 1):
            if len(row) != len(header):
                raise InputError(f"В записи {len(row)} полей, ожидалось {len(header)}", number)
            records.append(row)
    except csv.Error as exc:
        raise InputError(str(exc), len(records) + 1) from exc
    frame = pd.DataFrame(records, columns=header, dtype="string").loc[:, list(expected)]
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="record")
    return frame


def load(audit, directory, names, allow_empty):
    for name in names:
        spec = TABLES[name]
        path = directory / f"{name}.csv"
        try:
            raw = path.read_bytes()
            audit.sources[name] = hashlib.sha256(raw).hexdigest()
            original = read_csv(raw, spec.columns)
        except (OSError, UnicodeError, csv.Error, InputError) as exc:
            audit.log("input_unreadable", name, grain="table", record=getattr(exc, "record", None),
                      reason=exc.strerror if isinstance(exc, OSError) else str(exc))
            continue
        frame = original.copy()
        for col in frame.columns.difference(["text"]):
            frame[col] = original[col].str.strip()
        audit.tables[name] = frame
        if frame.empty and name not in allow_empty:
            audit.log("empty_table", name, grain="table")
        for col in frame.columns:
            audit.mark("trimmed_field", name, original[col].ne(frame[col]),
                       field=col)
        audit.mark("exact_duplicate", name, original.duplicated(keep=False))
        audit.profiles[name] = {"rows": len(frame), "columns": len(frame.columns)}


def check_contract(audit, name):
    spec, frame = TABLES[name], audit.tables[name]
    key = list(spec.key)
    market = name == "market_context"
    blank = pd.DataFrame({column: frame[column].str.strip().eq("") for column in frame}, index=frame.index)
    key_ok = ~blank[key].any(axis=1)
    audit.mark("missing_field" if market else "missing_key", name, ~key_ok, field=", ".join(key))
    audit.mark("benchmark_key" if market else "duplicate_key",
               name, key_ok & frame.duplicated(key, keep=False))
    profile = audit.profiles[name]
    profile["key_count"] = len(frame.loc[key_ok, key].drop_duplicates())
    profile["blank"] = {col: int(blank[col].sum()) for col in frame}
    profile["unique"] = {col: int(frame.loc[~blank[col], col].nunique()) for col in frame if col.endswith("_id")}
    for col in frame.columns.difference(key):
        if col == spec.date_column:
            continue
        rule = {
            "manager_id": "missing_manager", "q_pay": "missing_pay",
            "text": "missing_field",
        }.get(col, "missing_field")
        if (name in {"employees", "survey_responses"} and col in {"employee_id", "team_id", "division", "wave"}
                or not audit.annual_window and col == "wave"):
            rule = "missing_core_field"
        if col == "exit_type":
            rule = "missing_core_field"
        audit.mark(rule, name, blank[col], ("wave",) if col == "q_pay" else (), field=col)
    for col in frame.columns.intersection(ID_FORMATS):
        if col == "manager_id":
            continue
        optional = name == "comments" and col != "comment_id"
        pattern = r"T[0-9]{2,}" if col == "team_id" and not audit.annual_window else ID_FORMATS[col]
        audit.mark("invalid_optional_id" if optional else "invalid_id", name,
                   ~blank[col] & ~frame[col].str.fullmatch(pattern), (col,), field=col)
    for col in frame.columns.intersection(ENUMS):
        core = (name != "comments" and col in {"wave", "exit_type", "division"}
                or not audit.annual_window and col == "wave")
        allowed = audit.wave_dates if col == "wave" else ENUMS[col]
        if col == "division" and not audit.annual_window and "employees" in audit.tables:
            allowed = audit.tables["employees"]["division"]
        audit.mark("invalid_core_enum" if core else "invalid_enum", name,
                   ~blank[col] & ~frame[col].isin(allowed), (col,), field=col)


def parse_date(value, allow_dots):
    if allow_dots and re.fullmatch(r"[0-9]{2}\.[0-9]{2}\.[0-9]{4}", value):
        day, month, year = value.split(".")
        value = f"{year}-{month}-{day}"
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    return None


def check_date(audit, name):
    col = TABLES[name].date_column
    frame = audit.tables[name]
    comments = name == "comments"
    parsed = frame[col].map(lambda value: parse_date(value, comments)).astype("datetime64[us]")
    audit.dates[name] = parsed
    audit.mark("invalid_comment_date" if comments else "invalid_date",
               name, parsed.isna(), (col,), field=col)
    if comments:
        audit.mark("alternate_comment_date", name,
                   parsed.notna() & frame[col].str.contains(".", regex=False), (col,))
    valid = parsed.dropna()
    audit.profiles[name]["dates"] = [valid.min().date().isoformat(), valid.max().date().isoformat()] if len(valid) else []
    if "wave" in frame:
        expected = frame["wave"].map(audit.wave_dates).astype("datetime64[us]")
        months = parsed.dt.strftime("%Y-%m")
        valid_pair = expected.notna() & parsed.notna()
        audit.mark("comment_outside_wave_month" if comments else "outside_wave_month", name,
                   valid_pair & months.ne(frame["wave"]), ("wave", col))
        audit.mark("outside_wave_anchor", name,
                   valid_pair & months.eq(frame["wave"]) & parsed.ne(expected), ("wave", col))
        if not audit.annual_window:
            audit.mark("after_wave_cutoff", name, valid_pair & parsed.gt(expected), ("wave", col))


def check_pair_grain(audit, name, values):
    frame = audit.tables[name]
    repeated = frame.duplicated(PAIR, keep=False) & frame["employee_id"].ne("") & frame["wave"].ne("")
    key_name = TABLES[name].key[0]
    for (employee, wave), group in frame.loc[repeated].groupby(PAIR, sort=True):
        distinct = values.loc[group.index].nunique()
        differing = distinct[distinct.gt(1)].index.tolist()
        if name == "survey_responses":
            rule = "survey_pair_conflict" if differing else "survey_pair_duplicate"
        else:
            rule = "comment_pair_multiple"
        audit.log(rule, name, grain="employee_wave", key={"employee_id": employee, "wave": wave},
                  records=[int(index) for index in group.index], ids=group[key_name].tolist(),
                  differing_fields=differing)


def check_survey(audit):
    name = "survey_responses"
    frame = audit.tables[name]
    profile = audit.profiles[name]
    profile["scores"] = {}
    canonical = frame.loc[:, list(SCORES)].copy()
    for col in SCORES:
        lower = 0 if col == "q_recommend" else 1
        values = pd.to_numeric(frame[col], errors="coerce")
        in_range = values.between(lower, 10)
        valid = values.where(in_range).mod(1).eq(0).fillna(False)
        audit.mark("invalid_score", name, frame[col].ne("") & ~valid, ("wave", col), field=col)
        numbers = values[valid]
        canonical.loc[valid, col] = numbers.astype("int64").astype("string")
        profile["scores"][col] = {
            "lower": lower, "valid": len(numbers),
            "min": int(numbers.min()) if len(numbers) else None,
            "max": int(numbers.max()) if len(numbers) else None,
            "invalid": int((frame[col].ne("") & ~valid).sum()),
        }
    profile["pay_by_wave"] = frame["q_pay"].eq("").groupby(frame["wave"]).agg(["size", "sum"]).to_dict("index")
    check_pair_grain(audit, name, canonical)


def check_employees(audit):
    frame = audit.tables["employees"]
    managers = frame["manager_id"]
    invalid = ~managers.isin({*frame["employee_id"], "EMP00000"}) | managers.eq(frame["employee_id"])
    audit.mark("invalid_manager_reference", "employees", managers.ne("") & invalid, ("manager_id",))
    distinct = frame.loc[frame["division"].ne("")].groupby("team_id")["division"].nunique()
    audit.mark("team_multiple_divisions", "employees", frame["team_id"].isin(distinct[distinct > 1].index),
               ("team_id", "division"))
    dates = audit.dates["employees"]
    audit.mark("hire_after_snapshot", "employees", dates.gt(pd.Timestamp(audit.snapshot_end)),
               ("hire_date",))
    audit.diagnostics["external_managers"] = int(managers.eq("EMP00000").sum())


def check_exits(audit):
    dates = audit.dates["exits"]
    if not audit.annual_window:
        audit.mark("exit_out_of_window", "exits", dates.gt(pd.Timestamp(audit.snapshot_end)), ("exit_date",))
        return
    audit.mark("exit_out_of_window", "exits", dates.notna() & ~dates.between(pd.Timestamp(EXIT_START), pd.Timestamp(EXIT_END)),
               ("exit_date",))
    audit.mark("exit_outside_12_months", "exits", dates.between(pd.Timestamp(EXIT_START), pd.Timestamp(TRAILING_YEAR_START), inclusive="left"),
               ("exit_date",))
    audit.log("exit_window_length", "exits", grain="table", start=EXIT_START.isoformat(),
              end=EXIT_END.isoformat(), days=(EXIT_END - EXIT_START).days + 1,
              trailing_year_start=TRAILING_YEAR_START.isoformat())
    audit.diagnostics["exits_by_month"] = dict(sorted(Counter(d.isoformat()[:7] for d in dates.dropna()).items()))


def check_comments(audit):
    name = "comments"
    frame = audit.tables[name]
    text = frame["text"].str.strip()
    placeholder = text.str.fullmatch(r"(?:[\W_]+|нет комментариев[.!]?|без комментариев[.!]?|нет[.!]?|n/a)", case=False)
    audit.mark("comment_placeholder", name, placeholder, ("employee_id", "wave"))
    audit.profiles[name]["texts"] = int(text[text.ne("")].nunique())
    check_pair_grain(audit, name, text.to_frame())


def check_market(audit):
    name = "market_context"
    frame = audit.tables[name]
    decisions = []
    for record, row in frame.iterrows():
        key = {col: row[col] for col in TABLES[name].key}
        def log(rule, **details):
            audit.log(rule, name, record=int(record), key=key, **details)

        decision = "Требуется сверка определения."
        spec = BENCHMARKS.get(row["metric"])
        if spec is None:
            log("benchmark_metric")
        else:
            unit, lower, upper, method, decision = spec
            value = pd.to_numeric(row["value"], errors="coerce")
            if row["unit"] != unit or not lower <= value <= upper or value == float("inf"):
                log("benchmark_value", value=row["value"], unit=row["unit"], expected_unit=unit)
                decision = "Исключить до проверки значения и единицы."
            if row["methodology"] != method:
                log("benchmark_method", methodology=row["methodology"])
                decision = "Исключить из количественного сравнения до сверки методики и данных."
        period = row["period"]
        if not re.fullmatch(r"[0-9]{4}", period) or not 0 < int(period) < EXIT_END.year:
            log("benchmark_period", period=period)
            decision = "Отложить до сверки периода и доступности."
        elif int(period) < EXIT_END.year - 1:
            log("benchmark_period", period=period)
            decision = "Архив, только справочно." if spec and row["methodology"] == spec[3] else "Архив с неподтверждённой методикой. Исключить из сравнения."
        decisions.append((int(record), decision))
    if len(frame):
        audit.log("benchmark_publication", name, grain="table", affected_records=len(frame))
    audit.diagnostics["market"] = decisions


def run_audit(directory: Path, *, wave_date: date | None = None, snapshot_end: date | None = None) -> Audit:
    from .relationships import check_relationships

    if wave_date is None:
        audit = Audit()
        load(audit, directory, TABLES, ())
    else:
        cutoff = snapshot_end or wave_date
        if cutoff < wave_date:
            raise ValueError("The employment snapshot cannot precede the wave cutoff")
        audit = Audit(wave_dates={wave_date.isoformat()[:7]: wave_date}, snapshot_end=cutoff, annual_window=False)
        load(audit, directory, ("employees", "survey_responses", "comments", "exits"), {"exits", "comments", "survey_responses"})
    checks = {
        "employees": check_employees, "survey_responses": check_survey, "comments": check_comments,
        "exits": check_exits, "market_context": check_market,
    }
    for name in audit.tables:
        check_contract(audit, name)
        if TABLES[name].date_column:
            check_date(audit, name)
        checks[name](audit)
    check_relationships(audit)
    return audit
