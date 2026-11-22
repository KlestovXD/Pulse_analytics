from pathlib import Path

from src.pulse_analysis.report import count, link, number
from src.pulse_quality.report import section, table


ROOT = Path(__file__).resolve().parents[2]
FIELDS = {
    "sentiment": "Тональность", "severity": "Серьёзность", "urgency": "Срочность",
    "actionability": "Возможность действия", "quit_intent": "Намерение уйти",
    "objects": "Объект", "topics": "Темы", "ambiguity": "Неоднозначность",
}


def percent(value, digits=1):
    return "—" if value is None else number(100 * value, digits) + "%"


def metric(row):
    return "скрыто" if row["status"] != "available" else f"{count(row['comments'])} · {percent(row['share'])}"


def review_sections(lines, manifest, class_metrics, source):
    stats = manifest["review"]["substantive_agreement"]
    n = stats["texts"]
    categories = manifest["categorical_agreement"]
    fields = manifest["categorical_disagreements"]
    by_class = {(r["field"], r["category"]): r for r in class_metrics}
    section(lines, "Согласованность с проверенной человеком разметкой",
            "Контрольные оценки первоначально подготовили с помощью ИИ. Затем человек проверил и скорректировал их. "
            "Общая основа разметки могла повысить совпадение. Выборка целевая, каждый из 29 содержательных текстов "
            "учитывается один раз независимо от его частоты в выгрузке.",
            table(("Признак", "Совпали", "Совпадение", "κ Коэна"), (
                (FIELDS[key], f"{r['matches']}/{n}", percent(r["accuracy"]),
                 "—" if r["kappa"] is None else number(r["kappa"], 3))
                for key, r in categories.items()
            )),
            "Для серьёзности использована линейно взвешенная κ. Остальные κ номинальные. Для намерения уйти "
            "κ не определена, поскольку обе стороны выбрали одну категорию для всех текстов.",
            table(("Метрика определения тем", "Результат"), (
                ("Точность, микро", percent(stats["topics_micro_precision"])),
                ("Полнота, микро", percent(stats["topics_micro_recall"])),
                ("F1, микро", percent(stats["topics_micro_f1"])),
                ("Полное совпадение набора тем", percent(stats["topics_exact_match"])),
                ("Совпавшие / лишние / пропущенные тематические метки",
                 f"{stats['topics_shared']} / {stats['topics_extra']} / {stats['topics_missed']}"),
            )),
            f"Полнота обнаружения высоких и критических проблем составила {percent(stats['serious_problem_recall'])} "
            f"при {stats['serious_problems_in_reference']} положительных примерах. Ложных сигналов ухода "
            f"{sum(stats['unsupported_quit_signals'].values())} из {stats['no_quit_signal_in_reference']} "
            "текстов без признаков ухода. Чувствительность к прямым и косвенным сигналам ухода оценить нельзя.",
            table(("Признак", "Класс без примеров в проверенной разметке", "Предсказаний Gemini", "Не оценено"), (
                (FIELDS[r["field"]], r["category"], r["gemini_n"],
                 "полнота и F1" + (", точность" if not r["gemini_n"] else ""))
                for r in class_metrics if r["human_n"] == 0
            )),
            source("review_confusion.csv", "Матрицы ошибок") + " · " +
            source("review_class_metrics.csv", "Метрики по каждому классу") + ".")
    clarification = by_class["actionability", "требуется уточнение"]
    sarcasm = by_class["ambiguity", "сарказм"]
    context = by_class["ambiguity", "недостаток контекста"]
    section(lines, "Характерные расхождения",
            f"Чаще всего расходятся объект высказывания ({fields.get('objects', 0)} текстов), возможность "
            f"действия ({fields.get('actionability', 0)}) и неоднозначность ({fields.get('ambiguity', 0)}). "
            f"Gemini пропустила {clarification['fn']} из {clarification['human_n']} отметок «требуется уточнение» и "
            f"{context['fn']} из {context['human_n']} отметок о недостатке контекста.",
            f"Лишних тем {stats['topics_extra']}, пропущенных {stats['topics_missed']}. Например, к жалобе на "
            "долгий доступ модель добавила тему адаптации без указания на статус новичка. "
            f"Распознавание сарказма — {sarcasm['tp']}/{sarcasm['human_n']}. В ироничном комментарии "
            "о признании через новые задачи Gemini добавила тему нагрузки.",
            f"Серьёзность занижена в {stats['severity_direction']['lower']} случаях. Два расхождения касаются "
            "длительных переработок и невозможности восстановиться. Совпадение тональности скрывает эти различия.",
            source("review_disagreements.csv", "Все расхождения по полям с исходными текстами") +
            ". Внутренняя таблица контроля по шаблонам не раскрывает авторов, волны или подразделения. "
            "Выбор разных дословных фрагментов отмечен отдельно от расхождений в категориях.")


def render(manifest, class_metrics, volumes, distributions, scores, turnover, output, report):
    pipeline, review = manifest["pipeline"], manifest["review"]
    stats = review["substantive_agreement"]
    source = lambda name, label: link(report, output / name, label)
    indexed = {(r["level"], r["wave"], r["division"], r["dimension"], r["category"]): r for r in distributions}
    overall = next(r for r in volumes if r["level"] == "all")
    waves = [r for r in volumes if r["level"] == "wave"]
    divisions = [r for r in volumes if r["level"] == "division"]
    problems = sorted(
        (r for r in distributions if r["level"] == "all" and r["dimension"] == "topic_in_problem_comment" and r["comments"]),
        key=lambda r: (-r["comments"], r["category"]),
    )
    high = indexed["all", "", "", "severity", "высокая"]
    critical = indexed["all", "", "", "severity", "критическая"]
    lines = [
        "# «Пульс и риск текучести». Анализ открытых комментариев", "",
        f"Тональность совпала на **{manifest['categorical_agreement']['sentiment']['matches']}/{stats['texts']}** "
        f"проверенных текстах. Микро-F1 по темам составила **{percent(stats['topics_micro_f1'])}**. "
        f"Полнота обнаружения высоких и критических проблем — **{percent(stats['serious_problem_recall'])}** "
        f"при {stats['serious_problems_in_reference']} случаях в проверенной разметке. "
        "Определение серьёзности требует ручной сверки перед приоритизацией обращений.", "",
        f"Размечены **{count(overall['substantive_comments'])} содержательных комментариев**. "
        f"В них **{count(overall['substantive_confirmed_authors'])} подтверждённых авторов**. "
        "В отрицательных и смешанных записях чаще всего встречались оплата и нагрузка. "
        "Упоминания обеих тем сопровождались более низкими баллами по соответствующим вопросам внутри группы авторов.", "",
    ]

    section(lines, "Полнота и сверка источников",
            f"Уникальные содержательные тексты с валидной меткой — {pipeline['unique_labeled_texts']}. Проверено соответствие "
            f"меток, идентификаторов и хэшей всех {count(pipeline['comments'])} исходных записей. "
            f"Модель `{', '.join(pipeline['model_versions'])}`, инструкция `{pipeline['instruction_version']}`.",
            f"Из исходных записей {pipeline['row_status'].get('placeholder', 0)} — заглушки. "
            f"Контрольная проверка исключения заглушек — {review['placeholder_exclusion']['correctly_excluded']}/"
            f"{review['placeholder_exclusion']['texts_in_reference']}. "
            f"У {pipeline['unconfirmed_author_comments']} содержательных комментариев нет подтверждённого автора. "
            "Они учитываются в объёме и долях по волнам. Счётчики подтверждённых авторов их исключают. "
            f"Для разрезов по дивизионам использовано {count(pipeline['usable_org_comments'])} записей "
            "с проверенной привязкой. Правила подготовки также исключают отдельную запись в день увольнения.",
            f"Техническая очередь ручной проверки пуста. Признак `needs_review` в самой метке "
            f"присутствует у {pipeline['model_review_flags']} уникальных текстов.",
            source("analysis_manifest.json", "Детали сверки и контрольные суммы входов") + ".")
    review_sections(lines, manifest, class_metrics, source)

    section(lines, "Распределения и динамика",
            table(("Период", "Все записи", "Содержательные", "Подтверждённые авторы содержательных",
                   "Отрицательные", "Средняя серьёзность"), (
                (r["wave"] or "Все волны", count(r["comments"]), count(r["substantive_comments"]),
                 count(r["substantive_confirmed_authors"]),
                 metric(indexed[r["level"], r["wave"], "", "sentiment", "отрицательная"]),
                 metric(indexed[r["level"], r["wave"], "", "severity", "средняя"]))
                for r in [overall, *waves]
            )),
            "Доля считается от содержательных комментариев. Один автор может оставить несколько записей и "
            "участвовать в нескольких волнах. Доля отрицательных записей характеризует тексты в выгрузке.",
            table(("Тема в отрицательных или смешанных записях", "Все волны", "Подтверждённые авторы",
                   *(r["wave"] for r in waves)), (
                (r["category"], metric(r), count(r["unique_confirmed_authors"]),
                 *(metric(indexed["wave", w["wave"], "", r["dimension"], r["category"]]) for w in waves))
                for r in problems[:6]
            )),
            f"Пометка высокой серьёзности по всем волнам — {metric(high)}, критической — {metric(critical)}. "
            f"За высокой серьёзностью стоит {pipeline['unique_severity'].get('высокая', 0)} уникальный шаблон. "
            "Показатели отражают метки Gemini и чувствительны к выявленному занижению серьёзности. "
            "Малочисленные категории по отдельным волнам скрыты.",
            table(("Дивизион", "Содержательные", "Подтверждённые авторы", "Отрицательные",
                   "Оплата в отрицательных или смешанных", "Нагрузка в отрицательных или смешанных"), (
                (r["division"], count(r["substantive_comments"]), count(r["substantive_confirmed_authors"]),
                 metric(indexed["division", "", r["division"], "sentiment", "отрицательная"]),
                 metric(indexed["division", "", r["division"], "topic_in_problem_comment", "оплата"]),
                 metric(indexed["division", "", r["division"], "topic_in_problem_comment", "нагрузка"]))
                for r in divisions
            )),
            "Темы многозначны. Общая тональность относится ко всему комментарию, поэтому тема в смешанной записи "
            "может относиться к положительной части. Детали скрыты для групп и непустых категорий с менее чем "
            "10 подтверждёнными авторами. Для взаимоисключающих категорий скрывается дополнительная ячейка, "
            "если сумма раскрывает единственный скрытый ненулевой счётчик.",
            source("comment_volume.csv", "Объём по волнам и дивизионам") + " · " +
            source("comment_distribution.csv", "Все распределения, включая дивизион × волну") + ".")

    matched = [r for r in scores if not r["wave"] and r["comparison"] == "problem_comment" and r["status"] == "available"]
    section(lines, "Сопоставление с баллами опроса",
            f"Сопоставлено {count(pipeline['scored_linked_comments'])} комментариев в "
            f"{count(pipeline['scored_linked_employee_waves'])} парах «сотрудник × волна». Сравниваются авторы "
            "с упоминанием темы в отрицательной или смешанной записи и авторы без упоминаний темы. "
            "Для каждого сравнения учитываются только валидные оценки. Нагрузке сопоставлен вопрос о балансе работы "
            "и жизни. У адаптации и категории «другое» отдельных вопросов нет.",
            table(("Тема", "Вопрос", "Пар с темой", "Пар без темы", "С темой", "Без темы", "Разница"), (
                (r["topic"], r["question"], count(r["mentions_employee_waves"]), count(r["other_employee_waves"]),
                 number(r["mentions_mean"], 2), number(r["other_mean"], 2), number(r["difference"], 2, signed=True))
                for r in sorted(matched, key=lambda r: r["difference"])
            )),
            "Это описательные различия среди авторов. В общем срезе один сотрудник может участвовать в нескольких "
            "волнах. Представлены также сравнения по волнам и по любому упоминанию темы. "
            + source("topic_survey_links.csv", "Тематические оценки и численности") + ".")

    by_90 = [r for r in turnover if r["measure"] == "exit_90d"]
    annual = [r for r in turnover if r["measure"] == "annual_turnover"]
    section(lines, "Сопоставление с текучестью",
            "Доля отрицательных записей рассчитана по комментариям участников полной 90-дневной когорты, "
            "оставленным к дате ответа. Доля уходов рассчитана по всем участникам этой когорты в дивизионе. "
            "Сентябрь участвует только в годовом сравнении.",
            table(("Волна", "Дивизион", "Авторов", "Комментариев", "Отрицательные",
                   "Уходы / участники", "Доля ухода"), (
                (r["wave"], r["division"], count(r["unique_comment_authors"]), count(r["substantive_comments"]),
                 percent(r["negative_share"]), f"{r['exits']}/{r['turnover_denominator']}", percent(r["exit_rate"], 2))
                for r in by_90
            )),
            table(("Дивизион", "Отрицательные записи по трём волнам", "Годовая текучесть 01.10.2025–30.09.2026"), (
                (r["division"], percent(r["negative_share"]), percent(r["exit_rate"], 2)) for r in annual
            )),
            "В июне у функции HR выше доля последующего ухода, чем в розничном блоке, при меньшей доле "
            "отрицательных записей. Годовая текучесть относится ко всему персоналу дивизиона. "
            "Обе сверки описывают немногочисленные группы с разным составом и знаменателями. "
            "Выводы об индивидуальном риске и причинных эффектах на них не основаны.",
            source("comment_turnover_links.csv", "Групповые сопоставления с текучестью") + ".")
    section(lines, "Ограничения и воспроизведение",
            "Частые синтетические шаблоны повторяются. Точность тональности на целевой проверке не оценивает перенос "
            "на новые формулировки. Мало положительных примеров серьёзных проблем и нет примеров с сигналами ухода. "
            "Для новых комментариев нужна отдельная проверка с достаточным числом таких случаев.",
            "```powershell\nuv run --no-sync python -m src.pulse_comments analyze --review "
            "data/processed/comment_analysis/manual_review_filled.csv\n```",
            link(report, ROOT / "reports" / "quantitative_analysis.md", "Количественный отчёт и окна наблюдения") + ".")
    return "\n".join(lines)
