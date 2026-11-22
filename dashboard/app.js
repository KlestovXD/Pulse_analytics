"use strict";

const data = JSON.parse(document.getElementById("snapshot").textContent);
const byId = (id) => document.getElementById(id);
const el = (tag, content, className = "") => {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (className) node.className = className;
  return node;
};
const waves = data.waves;
const waveById = new Map(waves.map((wave) => [wave.id, wave]));
const metricInfo = {
  engagement: { label: "Вовлечённость", unit: "балла", axisUnit: "баллы" },
  participation: { label: "Участие", unit: "%", axisUnit: "%" },
  recommend_signal: { label: "eNPS", unit: "пункта", axisUnit: "пункты" },
};
const toneLabels = {
  "положительная": ["Положительная", "positive"],
  "отрицательная": ["Отрицательная", "negative"],
  "нейтральная": ["Нейтральная", "neutral"],
  "смешанная": ["Смешанная", "mixed"],
  "неопределенная": ["Неопределённая", "unclear"],
};
const monthFormat = new Intl.DateTimeFormat("ru-RU", { month: "long", timeZone: "UTC" });
const shortFormat = new Intl.DateTimeFormat("ru-RU", { month: "short", timeZone: "UTC" });
const num = (value, digits = 1) => value == null ? "—" :
  new Intl.NumberFormat("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits })
    .format(Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value).replace("-", "−");
const sign = (value, digits = 1) => (value >= 0.5 * 10 ** -digits ? "+" : "") + num(value, digits);
const dateLabel = (iso) => iso.split("-").reverse().join(".");
const getMonth = (id, formatter = monthFormat) => formatter.format(new Date(waveById.get(id).date + "T00:00:00Z"));
const capital = (label) => label.charAt(0).toUpperCase() + label.slice(1);
const state = { wave: waves.at(-1).id, division: "", metric: "engagement", team: "", view: "overview" };

function text(id, value) {
  byId(id).textContent = value;
}

function selection(control, options, selected) {
  control.replaceChildren(...options.map(([name, label]) => {
    const item = el("option", label);
    item.value = name;
    return item;
  }));
  control.value = selected;
}

function valueWithUnit(id, value, unit, digits) {
  const valueNode = byId(id);
  valueNode.replaceChildren(document.createTextNode(num(value, digits)));
  if (value != null) valueNode.append(el("small", unit));
}

function surveyScope() {
  return state.view === "overview" ? "" : state.division;
}

function surveyAt(wave, scope = surveyScope()) {
  return data.survey[scope]?.[wave];
}

function displayKpis() {
  const index = waves.findIndex((wave) => wave.id === state.wave);
  const current = surveyAt(state.wave);
  const earlier = index > 0 ? surveyAt(waves[index - 1].id) : null;
  for (const key of ["participation", "engagement", "recommend_signal"]) {
    const unit = metricInfo[key].unit;
    const value = current?.[key]?.[0];
    const previous = earlier?.[key]?.[0];
    valueWithUnit(key + "-value", value, unit, 1);
    const baseline = previous != null && value != null ?
      sign(value - previous) + (key === "participation" ? " п.п." : " п.") + " к прошлой волне" : "Первая доступная волна";
    text(key + "-detail", value == null ? "В выбранном срезе нет оценки" : baseline);
    text(key + "-period", capital(getMonth(state.wave)) + " · " + dateLabel(waveById.get(state.wave).date));
  }
  valueWithUnit("annual-value", data.annual.value[surveyScope()], "%", 2);
  const voluntary = data.annual.voluntary[surveyScope()];
  text("annual-voluntary", voluntary == null ? "Добровольная · —" : "Добровольная · " + num(voluntary, 2) + "%");
  text("annual-period", dateLabel(data.annual.start) + "–" + dateLabel(data.annual.end));
}

function svgElement(tag, attrs, label) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (label !== undefined) node.textContent = label;
  return node;
}

function trendChart() {
  const key = state.metric;
  const history = waves.filter((wave) => wave.id <= state.wave);
  const plotted = history.map((wave) => surveyAt(wave.id, "")?.[key] || null);
  const available = plotted.filter(Boolean);
  const place = byId("trend");
  text("trend-scope", "Весь блок · " + metricInfo[key].axisUnit);
  if (!available.length) {
    place.replaceChildren(el("p", "По этому срезу нет опубликованных оценок.", "empty"));
    return;
  }
  const values = available.map(([value]) => value);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const step = Math.max(1, Math.ceil((high - low) / 4));
  const start = Math.floor((low - step / 2) / step) * step;
  const end = Math.max(start + 3 * step, Math.ceil((high + step / 2) / step) * step);
  const w = 624, h = 124, top = 16, bottom = 95, left = 44, right = 592;
  const svg = svgElement("svg", { viewBox: "0 0 " + w + " " + h, role: "img",
    "aria-label": metricInfo[key].label + " по волнам. " +
      history.map((wave, index) => getMonth(wave.id) + " " + num(plotted[index]?.[0])).join(", ") });
  const y = (value) => bottom - (value - start) / (end - start) * (bottom - top);
  const x = (index) => history.length === 1 ? (left + right) / 2 :
    left + 38 + index / (history.length - 1) * (right - left - 76);
  for (let tick = start; tick <= end; tick += step) {
    svg.append(svgElement("line", { x1: left, x2: right, y1: y(tick), y2: y(tick), class: "grid" }));
    svg.append(svgElement("text", { x: 34, y: y(tick) + 4, "text-anchor": "end", class: "axis" }, num(tick, 0)));
  }
  let path = "", previous = false;
  plotted.forEach((point, index) => {
    if (point) path += (previous ? "L" : "M") + x(index) + " " + y(point[0]) + " ";
    previous = Boolean(point);
  });
  svg.append(svgElement("path", { d: path, class: "trend-line" }));
  plotted.forEach((point, index) => {
    const wave = history[index].id;
    const xPos = x(index);
    svg.append(svgElement("text", { x: xPos, y: 116, "text-anchor": "middle", class: "axis" },
      capital(getMonth(wave, shortFormat))));
    if (!point) return;
    const [value] = point;
    const marker = svgElement("circle", { cx: xPos, cy: y(value), r: 5,
      class: wave === state.wave ? "point current" : "point" });
    marker.append(svgElement("title", {}, capital(getMonth(wave)) + " · " + num(value)));
    svg.append(marker);
    svg.append(svgElement("text", { x: xPos + 12, y: y(value) - 8, class: "value-label" }, num(value)));
  });
  place.replaceChildren(svg);
}

function showPaired() {
  const first = surveyAt(waves[0].id)?.engagement?.[0];
  const current = surveyAt(state.wave)?.engagement?.[0];
  const pair = data.paired[surveyScope()]?.[state.wave];
  if (!pair || first == null || current == null) {
    const message = "Парное сравнение с первой волной недоступно.";
    text("panel-note", message);
    text("method-panel-note", message);
    return;
  }
  const comparison = capital(getMonth(waves[0].id)) + " — " + getMonth(state.wave) + ". Вовлечённость " +
    sign(current - first, 2) + " балла по всем ответам, " + sign(pair.delta, 2) + " у тех же участников";
  text("panel-note", comparison + ".");
  text("method-panel-note", comparison + " (95% ДИ " + num(pair.lo, 2) + "… " + num(pair.hi, 2) + ").");
}

function teamDetails() {
  const team = data.teams.find((item) => item.id === state.team);
  const details = byId("team-metrics");
  if (!team) {
    details.replaceChildren(el("p", "В выбранном дивизионе нет команд.", "empty"));
    return;
  }
  const row = team.waves[state.wave];
  if (!row) {
    details.replaceChildren(el("p", "Детали скрыты по правилу защиты малых выборок.", "empty"));
    return;
  }
  const cards = el("dl", undefined, "team-values");
  for (const metric of ["engagement", "participation", "recommend_signal"]) {
    const field = el("div");
    field.append(el("dt", metricInfo[metric].label), el("dd", num(row[metric][0]) +
      (metric === "participation" ? "%" : " п.")));
    cards.append(field);
  }
  const prior = waves[waves.findIndex((w) => w.id === state.wave) - 1];
  const old = prior ? team.waves[prior.id] : null;
  const meta = "Ответили " + row.respondents + " из " + row.active +
    (old ? " · Вовлечённость " + sign(row.engagement[0] - old.engagement[0]) + " п. к прошлой волне" : "");
  details.replaceChildren(cards, el("p", meta, "team-meta"));
}

function teamButton(id, label, className, action) {
  const button = el("button", undefined, className);
  button.type = "button";
  button.dataset.teamId = id;
  button.append(el("strong", id));
  if (label) button.append(el("span", label));
  button.addEventListener("click", action);
  return button;
}

function inspectTeam(teamId, wave) {
  const team = data.teams.find((item) => item.id === teamId);
  state.team = team.id;
  state.wave = wave;
  state.division = team.division;
  setView("divisions");
  byId("team").focus();
}

function attentionView() {
  const retail = "Розничный блок";
  const retailTurnover = data.annual.value[retail];
  const annual = data.annual.value[""];
  const target = byId("retail-focus");
  target.replaceChildren();
  if (retailTurnover != null && annual != null) {
    const button = el("button", retail, "text-button");
    button.type = "button";
    button.addEventListener("click", () => {
      state.division = retail;
      state.team = "";
      setView("divisions");
      document.querySelector("tr.selected button")?.focus();
    });
    target.append(button, el("span", num(retailTurnover, 2) + "% против " + num(annual, 2) + "% по блоку"));
  } else {
    target.append(el("span", "Сравнение недоступно.", "caption"));
  }
  const baseline = surveyAt(state.wave, "");
  const low = data.teams.filter((team) => {
    const row = team.waves[state.wave];
    return row && baseline && ["engagement", "recommend_signal"].every((key) =>
      baseline[key] && row[key][0] < baseline[key][0]);
  }).sort((a, b) => a.waves[state.wave].engagement[0] - b.waves[state.wave].engagement[0]).slice(0, 3);
  const levels = byId("adverse-teams");
  levels.replaceChildren(...low.map((team) => {
    const row = team.waves[state.wave];
    const scores = "Вовлечённость " + num(row.engagement[0]) + " · eNPS " + num(row.recommend_signal[0]);
    const button = teamButton(team.id, "", "team-chip", () => inspectTeam(team.id, state.wave));
    button.title = scores;
    button.setAttribute("aria-label", team.id + " · " + scores);
    return button;
  }));
  levels.append(el("span", low.length ? "Вовлечённость и eNPS" : "Доступных команд по этому критерию нет.", "muted"));

  const current = data.signals.filter((item) => item.wave === state.wave)
    .sort((a, b) => Number(b.confirmed) - Number(a.confirmed) || a.delta - b.delta);
  const earlier = data.signals.filter((item) => item.confirmed && item.wave < state.wave)
    .sort((a, b) => b.wave.localeCompare(a.wave) || a.q - b.q);
  const makeSignal = (item) => teamButton(item.team, sign(item.delta) + " п.",
    "signal team-chip" + (item.confirmed ? " confirmed" : ""),
    () => inspectTeam(item.team, item.wave));
  const signals = byId("signals");
  signals.replaceChildren();
  for (const confirmed of [true, false]) {
    const group = current.filter((item) => item.confirmed === confirmed);
    if (group.length) signals.append(...group.map(makeSignal),
      el("span", confirmed ? "Выдержало поправку" : "Предварительные · без подтверждения", "muted"));
  }
  if (!current.length) signals.append(el("span", state.wave === waves[0].id ?
    "Для первой волны предшествующего сравнения нет." : "В этой волне выделенных снижений нет.", "muted"));
  const previous = byId("previous-signal");
  previous.replaceChildren();
  if (earlier.length) {
    const last = earlier[0];
    previous.append(makeSignal(last), el("span", "Ранее после поправки · " +
      getMonth(last.previous) + " — " + getMonth(last.wave), "muted"));
  }
}

function showTeams() {
  const scope = data.teams.filter((team) => !state.division || team.division === state.division);
  const visibleTeams = new Set(scope.map((team) => team.id));
  const signals = data.signals.filter((item) => item.wave === state.wave && visibleTeams.has(item.team))
    .sort((a, b) => Number(b.confirmed) - Number(a.confirmed) || a.delta - b.delta);
  byId("detail-signals").replaceChildren(...signals.map((item) => teamButton(item.team,
    (item.confirmed ? "После поправки" : "Предварительный") + " · " +
    sign(item.delta, 2) + " балла [" + num(item.lo, 2) + "… " + num(item.hi, 2) + "], q " + num(item.q, 3) +
    " · пар " + item.n, "detail-signal", () => {
      state.team = item.team;
      byId("team").value = state.team;
      teamDetails();
    })));
  if (!signals.length) byId("detail-signals").append(el("p", "В данном срезе нет выделенных снижений.", "caption"));
  if (!visibleTeams.has(state.team)) state.team = signals[0]?.team ||
    scope.find((row) => row.waves[state.wave])?.id || scope[0]?.id || "";
  selection(byId("team"), scope.map((team) => [team.id, team.id]), state.team);
  byId("team").disabled = scope.length === 0;
  teamDetails();
}

function drawDivisions() {
  text("division-comparison-period", capital(getMonth(state.wave)) + " · " + dateLabel(waveById.get(state.wave).date));
  const list = data.divisions.map((division) => [division, surveyAt(state.wave, division)]);
  list.sort((a, b) => (b[1]?.engagement?.[0] ?? -Infinity) - (a[1]?.engagement?.[0] ?? -Infinity));
  const body = byId("division-rows");
  body.replaceChildren();
  for (const [division, metrics] of list) {
    const row = el("tr", undefined, division === state.division ? "selected" : "");
    const label = el("td");
    const button = el("button", division);
    button.type = "button";
    button.dataset.divisionName = division;
    button.setAttribute("aria-pressed", String(division === state.division));
    button.addEventListener("click", () => {
      state.division = division === state.division ? "" : division;
      state.team = "";
      byId("division").value = state.division;
      render();
    });
    label.append(button);
    row.append(label);
    const engagement = metrics?.engagement?.[0];
    const chart = el("td");
    const bar = el("div", undefined, "table-bar");
    if (engagement != null) {
      const track = el("i");
      const fill = el("i");
      fill.style.width = engagement + "%";
      track.setAttribute("aria-hidden", "true");
      track.append(fill);
      bar.append(track);
    }
    bar.append(el("span", num(engagement)));
    chart.append(bar);
    row.append(chart, el("td", num(metrics?.participation?.[0])), el("td", num(metrics?.recommend_signal?.[0])),
      el("td", num(data.annual.value[division], 2)));
    body.append(row);
  }
}

function tonesInto(target, slice) {
  const bar = el("div", undefined, "tone-bar");
  bar.setAttribute("aria-hidden", "true");
  const legend = el("ul", undefined, "tone-legend");
  for (const [name, [label, color]] of Object.entries(toneLabels)) {
    const share = slice.tone[name];
    if (share > 0) {
      const segment = el("span", undefined, color);
      segment.style.width = share + "%";
      bar.append(segment);
    }
    const item = el("li", undefined, share == null ? "masked-tone" : color);
    item.append(el("strong", share == null ? "Скрыто" : num(share) + "%"), el("span", label));
    legend.append(item);
  }
  target.replaceChildren(bar, legend);
}

function topicsInto(target, slice, limit = Infinity) {
  target.replaceChildren();
  for (const [key, title, color] of [
    ["positive_topics", "В положительных комментариях", "positive"],
    ["negative_topics", "В отрицательных комментариях", "negative"],
  ]) {
    const topics = Object.entries(slice[key]).filter(([, value]) => value > 0)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "ru")).slice(0, limit);
    const panel = el("div", undefined, "polarity " + color);
    panel.append(el("h3", title));
    const list = el("ul", undefined, "polarity-list");
    for (const [topic, share] of topics) {
      const row = el("li");
      row.append(el("span", capital(topic)), el("strong", num(share) + "%"));
      list.append(row);
    }
    panel.append(topics.length ? list : el("p", "Доступных тем нет.", "caption"));
    target.append(panel);
  }
}

function commentsView() {
  const overall = data.comments[state.wave][""];
  text("preview-comment-base", overall.n == null ? "База комментариев скрыта" :
    "Доли среди " + overall.n + " содержательных комментариев");
  tonesInto(byId("preview-tones"), overall);
  topicsInto(byId("preview-topics"), overall, 3);
  const slice = data.comments[state.wave][state.division];
  text("comments-scope", (state.division || "Весь блок") + " · " + (slice.n == null ?
    "Число записей скрыто" : "Содержательных записей " + slice.n));
  tonesInto(byId("tones"), slice);
  topicsInto(byId("topics"), slice);

  text("topic-score-scope", "По всем дивизионам выбранной волны · разница баллов среди авторов");
  const links = Object.entries(data.topic_differences[state.wave]).sort((a, b) => a[1] - b[1]);
  const score = byId("score-links");
  score.replaceChildren(...links.map(([name, value]) => {
    const row = el("div");
    row.append(el("span", capital(name) + (name === "нагрузка" ? " / баланс работы и жизни" : "")),
      el("strong", sign(value, 2)));
    return row;
  }));
  if (links.length) score.append(el("p", "Авторы, упомянувшие тему в отрицательной или смешанной записи, сравниваются с авторами без этой темы.", "caption"));
  else score.append(el("p", "Волна не даёт достаточной выборки для сопоставления.", "empty"));
}

function marketView() {
  const tbody = byId("market-rows");
  text("market-period", "Кейс " + dateLabel(data.annual.start) + "–" + dateLabel(data.annual.end) + " · банки за 2025 год");
  const percent = (value, local) => value == null ? "—" :
    num(value, local ? 2 : Number.isInteger(value) ? 0 : 1) + "%";
  for (const item of [
    { name: "Синтетический кейс", rates: { value: data.annual.value[""], voluntary: data.annual.voluntary[""] } },
    ...data.market,
  ]) {
    const row = el("tr", undefined, item.url ? "" : "case-rate");
    const label = el("th");
    label.scope = "row";
    if (item.url) {
      const link = el("a", item.name);
      link.href = item.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.title = "Публикация от " + dateLabel(item.published);
      label.append(link);
    } else label.textContent = item.name;
    row.append(label, el("td", percent(item.rates.value, !item.url)), el("td", percent(item.rates.voluntary, !item.url)));
    tbody.append(row);
  }
}

function setView(view) {
  state.view = view;
  document.querySelectorAll(".view").forEach((node) => { node.hidden = node.id !== state.view; });
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.view === state.view));
  });
  text("view-title", { overview: "Пульс сотрудников", divisions: "Подразделения и команды", comments: "Комментарии сотрудников" }[view]);
  byId("division-filter").hidden = view === "overview";
  document.querySelector(".kpis").hidden = view === "comments";
  byId("wave").value = state.wave;
  byId("division").value = state.division;
  render();
}

function render() {
  displayKpis();
  trendChart();
  showPaired();
  attentionView();
  showTeams();
  drawDivisions();
  commentsView();
}

selection(byId("wave"), waves.map((wave) => [wave.id, capital(getMonth(wave.id)) + " " + wave.id.slice(0, 4)]), state.wave);
selection(byId("division"), [["", "Весь блок"], ...data.divisions.map((value) => [value, value])], state.division);
byId("wave").addEventListener("change", (event) => { state.wave = event.target.value; render(); });
byId("division").addEventListener("change", (event) => { state.division = event.target.value; state.team = ""; render(); });
byId("team").addEventListener("change", (event) => { state.team = event.target.value; teamDetails(); });
document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
document.querySelectorAll("[data-metric]").forEach((button) => button.addEventListener("click", () => {
  state.metric = button.dataset.metric;
  document.querySelectorAll("[data-metric]").forEach((item) => {
    item.setAttribute("aria-pressed", String(item.dataset.metric === state.metric));
  });
  trendChart();
}));
byId("open-comments").addEventListener("click", () => setView("comments"));
byId("open-divisions").addEventListener("click", () => setView("divisions"));
byId("open-method").addEventListener("click", () => byId("method").showModal());
byId("close-method").addEventListener("click", () => byId("method").close());
const [oddsRatio, oddsLow, oddsHigh] = data.outcome_model;
text("method-model", oddsRatio == null ? "Оценка регрессии недоступна." :
  "Связь с увольнением в следующие 90 дней. Отношение шансов на дополнительные 10 баллов вовлечённости " +
  num(oddsRatio, 2) + " (95% ДИ " + num(oddsLow, 2) + "… " + num(oddsHigh, 2) + ") с учётом волны.");
marketView();
setView("overview");
