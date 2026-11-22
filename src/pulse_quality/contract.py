from dataclasses import dataclass
from datetime import date


SCORES = (
    "q_conditions", "q_recognition", "q_growth", "q_manager", "q_pay",
    "q_balance", "q_goals", "q_tools", "q_recommend",
)
WAVE_DATES = {
    "2026-03": date(2026, 3, 16),
    "2026-06": date(2026, 6, 15),
    "2026-09": date(2026, 9, 14),
}
EXIT_START = date(2025, 9, 1)
EXIT_END = date(2026, 9, 30)
TRAILING_YEAR_START = date(2025, 10, 1)
SMALL_GROUP = 10
PAIR = ["employee_id", "wave"]
ID_FORMATS = {
    "employee_id": r"EMP[0-9]{5}", "manager_id": r"EMP[0-9]{5}",
    "team_id": r"T(?:0[1-9]|[1-3][0-9]|40)", "response_id": r"R[0-9]{6}",
    "comment_id": r"C[0-9]{5}",
}
DIVISIONS = {
    "Розничный блок", "ИТ-платформа", "Корпоративный бизнес",
    "Операционный блок", "Функция HR",
}
ENUMS = {
    "wave": WAVE_DATES.keys(),
    "division": DIVISIONS,
    "grade": {"Специалист", "Ведущий специалист", "Главный специалист", "Руководитель группы"},
    "work_format": {"офис", "гибрид", "удалёнка"},
    "exit_type": {"добровольный", "вынужденный"},
}


@dataclass(frozen=True)
class Table:
    columns: tuple[str, ...]
    key: tuple[str, ...]
    date_column: str | None = None


TABLES = {
    "employees": Table(
        ("employee_id", "team_id", "division", "grade", "work_format", "hire_date", "manager_id"),
        ("employee_id",), "hire_date",
    ),
    "survey_responses": Table(
        ("response_id", "wave", "employee_id", "submitted_at", *SCORES),
        ("response_id",), "submitted_at",
    ),
    "comments": Table(
        ("comment_id", "wave", "employee_id", "team_id", "division", "submitted_at", "text"),
        ("comment_id",), "submitted_at",
    ),
    "exits": Table(("employee_id", "exit_date", "exit_reason", "exit_type"), ("employee_id",), "exit_date"),
    "market_context": Table(
        ("source", "metric", "scope", "period", "value", "unit", "methodology"),
        ("source", "metric", "scope", "period"),
    ),
}

# Для рыночной таблицы key — аналитический ключ сверки, а не первичный ключ источника.
BENCHMARKS = {
    "текучесть": ("%", 0, 100, "все увольнения / среднегодовая численность",
                 "Условно после согласования годового окна, средней численности и сегмента."),
    "текучесть добровольная": ("%", 0, 100, "только добровольные увольнения / среднегодовая численность",
                              "Условно после сверки типа увольнений, окна, знаменателя и сегмента."),
    "доля добровольных увольнений": ("%", 0, 100, "доля добровольных среди всех увольнений",
                                    "Условно как структура увольнений при совпадении окна и сегмента."),
    "eNPS": ("п.п.", -100, 100, "ежегодный опрос, вопрос о рекомендации работодателя",
             "Справочно. Нужны пороги, выборка и сопоставимый период ежегодного опроса."),
    "индекс вовлечённости": ("балл 0–100", 0, 100, "полный ежегодный опросник, нормировка на 100",
                            "Справочно. Пересчёт шкалы не согласует состав вопросов, веса и выборку."),
    "стоимость замещения": ("% годовой ЗП", 0, float("inf"), "подбор + адаптация + простой позиции",
                           "Сценарий. Для оценки нужны данные о годовых зарплатах и затратах."),
}
