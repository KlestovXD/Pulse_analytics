import os
from pathlib import Path

import pandas as pd

from src.pulse_quality.contract import WAVE_DATES
from src.pulse_quality.report import section, table

from .metrics import SCORE_METRICS
from .prepare import THEMES


ROOT = Path(__file__).resolve().parents[2]


def number(value, digits=1, signed=False):
    if pd.isna(value):
        return "—"
    value = 0 if abs(value) < 0.5 * 10**-digits else value
    pattern = f"{'+' if signed and value else ''}.{digits}f"
    return format(value, pattern).replace(".", ",").replace("-", "−")


def cell(value, low, high, digits=1, signed=False):
    if pd.isna(value):
        return "—"
    return f"{number(value, digits, signed)} [{number(low, digits)}…{number(high, digits)}]"


def estimate(row, metric, digits=1):
    return cell(row[metric], row[f"{metric}_lo"], row[f"{metric}_hi"], digits)


def count(value):
    return "<10" if pd.isna(value) else f"{int(value):,}".replace(",", " ")


def link(report, target, label):
    relative = os.path.relpath(target, report.parent).replace(os.sep, "/")
    return f"[{label}]({relative.replace(' ', '%20')})"


def wave_section(lines, overall, divisions):
    section(lines, "Все активные участники каждой волны",
            "Участие приведено в процентах. Индексы имеют шкалу 0–100, eNPS выражен в пунктах. "
            "В скобках указаны ориентировочные 95%-ные интервалы. Они описывают разброс оценок при принятых "
            "предпосылках и не учитывают систематический самоотбор в опрос.",
            table(("Волна", "Активные", "Ответившие", "Участие, %", "Вовлечённость", "n индекса",
                   "Индекс 8 тем", "n индекса 8 тем", "eNPS", "n eNPS"), (
                (wave, count(r["active"]), count(r["respondents"]), estimate(r, "participation"),
                 estimate(r, "engagement"), count(r["engagement_n"]), estimate(r, "state8"),
                 count(r["state8_n"]), estimate(r, "recommend_signal"), count(r["recommend_signal_n"]))
                for wave, r in overall.iterrows()
            )),
            "Восьмитемный индекс требует валидных ответов по всем восьми темам. Он имеет меньший знаменатель, "
            "поэтому различия с основным индексом отражают и охват вопросов, и состав доступных наблюдений.",
            "Четырёхтемный индекс среди участников с полными восемью оценками по волнам — "
            + ", ".join(f"{wave} · {number(r['engagement_same8'], 2)}" for wave, r in overall.iterrows()) + ".",
            table(("Тема, шкала 1–10", *overall.index), (
                (SCORE_METRICS[metric], *(f"{estimate(r, metric, 2)} (n={count(r[f'{metric}_n'])})"
                                         for _, r in overall.iterrows())) for metric in THEMES
            )),
            table(("Дивизион", "Волна", "Участие, %", "Вовлечённость", "Δ к прошлой волне",
                   "Индекс 8 тем", "eNPS", "Ответившие"), (
                (r["group"], r["wave"], number(r["participation"]), number(r["engagement"]),
                 number(r["engagement_delta"], signed=True), number(r["state8"]),
                 number(r["recommend_signal"]), count(r["respondents"])) for _, r in divisions.iterrows()
            )))


def composition_section(lines, paired, composition, decomposition, overall, processed, report):
    last_wave = overall.index[-1]
    latest = composition.loc[composition["wave"].eq(last_wave)]
    main_pairs = paired.loc[paired["level"].eq("total") & paired["metric"].isin(
        ["engagement", "state8", "recommend_signal"]
    )]
    section(lines, "Изменение оценок и состав участников",
            "В парном анализе входят сотрудники, ответившие в обе сравниваемые волны. "
            "Для каждой метрики используются пары с валидным значением в обеих волнах. "
            "В строках «Все три волны» сотрудник участвовал также в июне. Знак Δ соответствует поздней волне минус ранняя.",
            table(("Волны", "Выборка", "Метрика", "Валидные пары", "Ранняя", "Поздняя", "Δ [95% ДИ]"), (
                (f"{r.earlier} → {r.later}", "Все три волны" if r.cohort == "all_three" else "Обе волны",
                 SCORE_METRICS[r.metric], count(r.n), number(r.earlier_value), number(r.later_value),
                 cell(r.delta, r.lo, r.hi, 2, signed=True)) for r in main_pairs.itertuples()
            )),
            "Первое разложение сопоставляет общие средние и парную выборку. Ранний сдвиг равен "
            "среднему в парной выборке минус общее среднее в ранней волне. Поздний сдвиг равен общему среднему "
            "минус парное в поздней волне. В отдельном разложении общий сдвиг разбит на изменение оценок "
            "внутри дивизионов и изменение долей дивизионов. Ни одно из разложений не устанавливает причину изменения.",
            table(("Волны", "Общее Δ", "Парное Δ", "Ранний сдвиг", "Поздний сдвиг", "Внутри дивизионов",
                   "Доли дивизионов"), (
                (f"{r.earlier} → {r.later}", *(number(getattr(r, col), 3, signed=True)
                  for col in ("observed_delta", "paired_delta", "earlier_selection", "later_selection",
                              "within_division", "division_mix"))) for r in decomposition.itertuples()
            )),
            f"Состав участников в последней волне ({last_wave}). Разрыв представленности показывает долю "
            "среди ответивших минус долю среди активных. "
            + link(report, processed / "participant_composition.csv", "Все волны, численности и интервалы участия") + ".",
            table(("Разрез", "Группа", "Доля в штате, %", "Доля в ответах, %", "Разрыв, п.п.",
                   "Δ доли к прошлой волне, п.п."), (
                ({"division": "Дивизион", "grade": "Грейд", "work_format": "Формат работы"}[r.dimension], r.group,
                 number(r.active_share), number(r.respondent_share), number(r.representation_gap, signed=True),
                 number(r.respondent_share_change, signed=True)) for r in latest.itertuples()
            )))


def annual_section(lines, overall, annual):
    section(lines, "Годовая текучесть и чувствительность к дню увольнения",
            "Годовое окно 01.10.2025–30.09.2026. Текучесть равна числу выходов, делённому на среднюю ежедневную "
            "численность. В дополнительном сценарии сотрудники учитываются в численности в день увольнения. "
            "Числители годовой текучести при этом остаются прежними.",
            table(("Группа", "Увольнения", "Добровольные", "Ср. численность", "Общая, %", "Добровольная, %",
                   "Общая с днём ухода, %", "Добровольная с днём ухода, %"), (
                ("Все сотрудники" if r.level == "total" else r.group, count(r.all_exits), count(r.voluntary_exits),
                 number(r.mean_headcount, 2), number(r.turnover, 3), number(r.voluntary_turnover, 3),
                 number(r.turnover_inclusive, 3), number(r.voluntary_turnover_inclusive, 3))
                for r in annual.itertuples()
            )),
            table(("Волна", "Активные базовые / с днём ухода", "Ответившие базовые / с днём ухода",
                   "Участие базовое / с днём ухода, %", "Вовлечённость с днём ухода", "eNPS с днём ухода"), (
                (wave, f"{count(r['active'])} / {count(r['active_inclusive'])}",
                 f"{count(r['respondents'])} / {count(r['respondents_inclusive'])}",
                 f"{number(r['participation'], 2)} / {number(r['participation_inclusive'], 2)}",
                 number(r["engagement_inclusive"], 2), number(r["recommend_signal_inclusive"], 2))
                for wave, r in overall.iterrows()
            )))


def outcome_section(lines, risks, scores, models):
    labels = {"total": "Всего", "wave": "Волна", "division": "Дивизион", "grade": "Грейд",
              "work_format": "Формат работы", "recommend_category": "Ответ на рекомендацию"}
    by_wave = models.loc[models["model"].isin(("engagement_2026-03", "engagement_2026-06"))]
    conclusion = "Сравнивать величину связи между волнами следует с учётом ширины интервалов."
    if by_wave["odds_ratio"].lt(1).all() and by_wave["lo"].le(1).all() and by_wave["hi"].ge(1).all():
        conclusion = "Связь с вовлечённостью направлена одинаково в обеих волнах. Интервал каждой отдельной оценки включает 1."
    section(lines, "Связь с увольнением в течение 90 дней",
            "Когорта включает активных на дату ответа сотрудников марта и июня. Событие учитывается "
            "с первого по 90-й день после ответа. Волна сентября исключена из-за неполного горизонта. "
            "При нескольких версиях датой признаков служит самая поздняя дата, участвовавшая в сведении. "
            "Повторные наблюдения одного сотрудника учитываются кластеризацией по ID. "
            "Доля по каждой волне имеет интервал Уилсона, остальные доли используют бутстрэп по сотрудникам. "
            "На границах 0% и 100% применяется интервал Уилсона по числу уникальных сотрудников.",
            table(("Разрез", "Группа", "Наблюдения", "Сотрудники", "Увольнения", "Доля, % [95% ДИ]"), (
                (labels[r.dimension], "Март и июнь" if r.dimension == "total" else r.group,
                 count(r.observations), count(r.employees), count(r.exits), cell(r.risk_pct, r.lo, r.hi, 2))
                for r in risks.itertuples()
            )),
            "Разница ниже рассчитана как оценка ушедших минус оценка остальных в полном 90-дневном окне. "
            "Интервалы для разниц учитывают зависимость наблюдений внутри сотрудника. Сравнения описательные, "
            "с поправкой на множественность они не проверялись.",
            table(("Метрика", "Валидных с уходом", "Валидных без ухода", "С уходом", "Без ухода", "Разница [95% ДИ]"), (
                (SCORE_METRICS[r.metric], count(r.n_exits), count(r.n_no_exit), number(r.mean_exit, 2),
                 number(r.mean_no_exit, 2), cell(r.difference, r.lo, r.hi, 2, signed=True))
                for r in scores.itertuples()
            )),
            "Логистические модели используют по одному баллу за раз и индикатор волны для объединённой когорты. "
            "Интервалы для отношения шансов учитывают кластеризацию по сотруднику. Для eNPS индивидуальный "
            "балл равен −100, 0 или +100. Модель использует этот упорядоченный балл. "
            "Для сравнения двух индексов четырёхтемную модель также оценили на выборке с ответами на все восемь тем. "
            "Числа исходов ограничивают объём поправок на потенциально смешивающие факторы.",
            table(("Модель", "Шаг признака", "Наблюдения", "Увольнения", "Отношение шансов [95% ДИ]", "Сходимость"), (
                (f"{SCORE_METRICS[r.metric]} · " +
                 ("полные 8 тем" if r.model == "engagement_same8" else ("март и июнь" if r.wave_adjusted else r.model[-7:])),
                 f"+{r.unit} баллов", count(r.observations), count(r.exits), cell(r.odds_ratio, r.lo, r.hi, 3),
                 {"converged": "да", "insufficient_information": "мало данных",
                  "no_stable_fit": "неустойчива"}[r.status]) for r in models.itertuples()
            )),
            conclusion + " Оценки описывают связи внутри наблюдаемой выборки. Влияние отдельных HR-мер и индивидуальный риск "
            "по этим моделям оценить нельзя. Разрезы по дивизиону, грейду и формату работы не поправлены "
            "на остальные различия между группами.")


def teams_section(lines, teams, processed, report):
    suppressed = teams["publication_status"].eq("respondents_lt_10").groupby(teams["wave"]).sum()
    flags = teams.loc[teams["attention"].eq("paired_decline_signal")]
    sections = [
        "Сигнал для обсуждения — отрицательное изменение вовлечённости в сопоставимой парной выборке "
        "при ≥10 валидных парах, если верхняя граница ориентировочного 95%-ного интервала ниже нуля. "
        "Сигнал помогает выбрать команды для разбора. При одновременной проверке команд возможны случайные флаги.",
        table(("Волна", "Срезов с <10 ответившими"), ((wave, n) for wave, n in suppressed.items())),
        link(report, processed / "team_metrics.csv", "Все командные срезы и признаки неопределённости")
        + ". Годовая текучесть в таблице показана рядом с последней волной как отдельный итог на 30 сентября. "
        "Сигнал по командам опирается на доступные к дате волны ответы. Для малых групп скрыты численность, "
        "точное число ответивших, оценки, участие, изменения и годовая текучесть.",
    ]
    if len(flags):
        sections.append(table(("Команда", "Дивизион", "Волна", "Валидные пары", "Парное Δ [95% ДИ]"), (
            (r.team_id, r.division, r.wave, count(r.paired_engagement_n),
             cell(r.paired_engagement_delta, r.paired_engagement_lo, r.paired_engagement_hi, 2, signed=True))
            for r in flags.itertuples()
        )))
    else:
        sections.append("Команд с таким сигналом в доступных срезах нет.")
    section(lines, "Команды для разбора", *sections)


def market_section(lines, market, processed, report):
    peers = market.loc[market["source_type"].eq("bank_disclosure") & market["period_end"].eq("2025-12-31")
                       & market["publication_timing"].eq("by_cutoff") & market["absolute_difference"].notna()]
    scenarios = market.loc[market["comparison_mode"].eq("scenario") & market["absolute_difference"].notna()]
    section(lines, "Внешний контекст и сопоставимость",
            "Сравнение с опубликованными показателями отдельных банков за 2025 год условное. "
            "Внутренние показатели рассчитаны за 01.10.2025–30.09.2026. Правила формирования числителя "
            "и знаменателя требуют сверки по каждому банку.",
            table(("Источник", "Метрика", "Кейс, %", "Банк, %", "Разница, п.п."), (
                (f"[{r.source}]({r.source_url})", r.metric, number(r.internal_value, 2),
                 number(r.value, 1), number(r.absolute_difference, 2, signed=True))
                for r in peers.itertuples()
            )),
            "Опрос Antal Talent за 2025 год даёт ориентир для офисных сотрудников банков и финансовых институтов. "
            "Для сопоставления с кейсом нужна карта ролей и единая формула. Публичных банковских ориентиров "
            "для eNPS и четырёхкомпонентного индекса вовлечённости с подтверждённой общей методикой не найдено. "
            "Рыночное отклонение по дивизионам не оценено.",
            *(
                f"В синтетическом справочнике банковский ориентир за {r.period} год составляет {number(r.value)}%. "
                f"Разность с {number(r.internal_value, 2)}% равна "
                f"{number(r.absolute_difference, 2, signed=True)} п.п. "
                "Это сценарная арифметика по неподтверждённому источнику."
                for r in scenarios.itertuples()
            ),
            link(report, ROOT / "reports" / "market_analysis.md", "Выводы по рынку") + " · "
            + link(report, ROOT / "reports" / "market_sources.md", "Реестр публичных источников") + " · "
            + link(report, processed / "market_comparison.csv", "Итоговая таблица сопоставлений") + " · "
            + link(report, processed / "market_comparability.csv", "Исходная диагностика синтетических ориентиров") + ".")


def render(datasets, processed, report):
    frame = lambda name: datasets[name][0]
    waves = frame("wave_metrics")
    overall = waves.loc[waves["level"].eq("total")].set_index("wave")
    divisions = waves.loc[waves["level"].eq("division")]
    paired, decomposition = frame("paired_dynamics"), frame("engagement_decomposition")
    first, last = overall.index[[0, -1]]
    pair = paired.loc[paired["level"].eq("total") & paired["metric"].eq("engagement")
                      & paired["cohort"].eq("pair") & paired["earlier"].eq(first) & paired["later"].eq(last)].iloc[0]
    panel_note = (
        " Интервал парной оценки включает ноль. Рост среднего по всем участникам требует учёта состава ответивших."
        if pd.notna(pair["lo"]) and pair["lo"] <= 0 <= pair["hi"]
        and overall.loc[last, "engagement"] > overall.loc[first, "engagement"] else ""
    )
    totals = frame("annual_turnover").loc[lambda f: f["level"].eq("total")].iloc[0]
    risks, scores, models = frame("exit_risk_groups"), frame("exit_score_differences"), frame("exit_logistic_models")
    main_model = models.loc[models["model"].eq("engagement_adjusted")].iloc[0]
    lines = [
        "# «Пульс и риск текучести». Количественный анализ", "",
        f"С марта по сентябрь средняя вовлечённость среди участников изменилась с "
        f"**{number(overall.loc[first, 'engagement'], 2)} до {number(overall.loc[last, 'engagement'], 2)}**. "
        f"Изменение у сотрудников с валидной оценкой в обеих волнах составило "
        f"**{cell(pair['delta'], pair['lo'], pair['hi'], 2, signed=True)} балла** "
        f"при {count(pair['n'])} валидных парах.{panel_note} Участие по волнам изменилось с "
        f"{number(overall.loc[first, 'participation'])}% до {number(overall.loc[last, 'participation'])}%.", "",
        f"Общая годовая текучесть составила **{number(totals['turnover'], 2)}%**, добровольная — "
        f"**{number(totals['voluntary_turnover'], 2)}%**. Для мартовских и июньских участников отношение шансов "
        f"уйти в следующие 90 дней на дополнительные 10 баллов вовлечённости составило "
        f"**{cell(main_model['odds_ratio'], main_model['lo'], main_model['hi'], 2)}** после учёта волны.", "",
    ]
    wave_section(lines, overall, divisions)
    composition_section(lines, paired, frame("participant_composition"), decomposition, overall, processed, report)
    annual_section(lines, overall, frame("annual_turnover"))
    outcome_section(lines, risks, scores, models)
    teams_section(lines, frame("team_metrics"), processed, report)
    market_section(lines, frame("market_comparison"), processed, report)
    section(lines, "Ограничения и воспроизводимость",
            "- Состав команд восстановлен по единственному реестру. Истории переводов, изменений грейда и повторного "
            "найма в выгрузке нет. Даты волн взяты из аудита по наблюдаемой отправке.\n"
            "- Ответы добровольные. Интервалы предполагают наблюдаемую структуру данных и не покрывают "
            "систематический самоотбор, пропуски или искажения реестра.\n"
            "- Проверка связей и команд использует несколько сравнений. Сигналы требуют проверки на новых волнах.\n"
            "- Для годовой текучести доступны фактические события и экспозиция. Интервалы выборочной "
            "погрешности для этой описательной административной метрики не рассчитывались.",
            link(report, processed / "manifest.json", "Реестр таблиц, аналитические ключи и SHA-256 входов")
            + " · " + link(report, ROOT / "reports" / "data_quality.md", "Аудит источников") + " · "
            + link(report, ROOT / "docs" / "metric_registry.md", "Реестр метрик") + ".")
    return "\n".join(lines)
