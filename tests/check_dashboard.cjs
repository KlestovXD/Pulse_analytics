// Optional browser acceptance check. Provide a local Playwright installation and Chromium executable.
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const root = path.resolve(__dirname, "..");
const output = process.env.DASHBOARD_SCREENSHOTS;
const pageUrl = pathToFileURL(path.join(root, "dashboard", "index.html")).href;

async function verify(browser, viewport) {
  const context = await browser.newContext({ viewport, offline: true, locale: "ru-RU" });
  const page = await context.newPage();
  const pageErrors = [], externalRequests = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("request", (request) => {
    if (/^https?:/.test(request.url())) externalRequests.push(request.url());
  });
  await page.goto(pageUrl);
  assert.equal(await page.locator("#wave").inputValue(), "2026-09");
  assert.equal(await page.locator("#participation-value").textContent(), "74,3%");
  assert.match(await page.locator("#engagement-value").textContent(), /^62,6/);
  assert.match(await page.locator("#recommend_signal-value").textContent(), /^−9,9/);
  assert.equal(await page.locator("#annual-value").textContent(), "14,42%");
  assert.equal(await page.locator("#annual-period").textContent(), "01.10.2025–30.09.2026");
  assert.match(await page.locator("#annual-voluntary").textContent(), /13,12%/);
  assert.match(await page.locator("#panel-note").textContent(), /\+0,88.*−0,13/);
  assert.equal(await page.locator("#trend circle").count(), 3);
  assert.deepEqual(await page.locator("#signals .signal strong").allTextContents(), ["T06", "T34"]);
  assert.equal(await page.locator("#signals .signal.confirmed").count(), 0);
  assert.match(await page.locator("#previous-signal").textContent(), /T16/);
  assert.match(await page.locator("#retail-focus").textContent(), /Розничный блок.*20,19%.*14,42%/);
  assert.deepEqual(await page.locator("#adverse-teams button").allTextContents(), ["T07", "T03", "T35"]);
  assert.match(await page.locator("#preview-tones").textContent(), /50,1%Положительная39,1%Отрицательная/);
  assert.match(await page.locator("#preview-topics .positive").textContent(), /Развитие11,3%/);
  assert.match(await page.locator("#preview-topics .negative").textContent(), /Нагрузка11,3%/);
  assert.deepEqual(await page.locator("#market-rows td").allTextContents(), ["14,42%", "13,12%", "21,1%", "18,6%", "27%", "19%"]);
  assert.equal(await page.locator("#market-period").textContent(), "Кейс 01.10.2025–30.09.2026 · банки за 2025 год");
  assert.equal(await page.locator('#market-rows a[href^="https://"]').count(), 2);
  assert.match(await page.locator(".market-caption").textContent(), /Репрезентативного среднерыночного ориентира нет/);
  assert.doesNotMatch(await page.locator("#overview").innerText(), /95% ДИ|отношение шансов|q 0,/i);
  assert.equal(await page.locator("#division").isVisible(), false);
  assert.equal(await page.locator("#team").isVisible(), false);
  for (const selector of [".comment-preview", ".market-panel", ".attention-panel"]) {
    const bounds = await page.locator(selector).boundingBox();
    assert.ok(bounds && bounds.y + bounds.height <= viewport.height, selector + " extends past the viewport");
  }
  const dimensions = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  assert.equal(dimensions[0], dimensions[1], "horizontal overflow on overview");

  if (output) {
    fs.mkdirSync(output, { recursive: true });
    await page.screenshot({ path: path.join(output, "overview-" + viewport.width + ".png") });
  }

  await page.locator("[data-metric='participation']").click();
  assert.equal(await page.locator("#trend circle").count(), 3);
  assert.match(await page.locator("#trend").textContent(), /74,3/);
  await page.locator("[data-metric='recommend_signal']").click();
  assert.match(await page.locator("#trend").textContent(), /−9,9/);
  await page.locator("#signals [data-team-id='T34']").click();
  assert.equal(await page.locator("#divisions").isVisible(), true);
  assert.equal(await page.locator("#team").inputValue(), "T34");
  assert.equal(await page.locator("#team-metrics dd").count(), 3);
  assert.match(await page.locator("#detail-signals").textContent(), /q 0,637/);
  await page.locator("#division").selectOption("");
  await page.locator("#team").selectOption("T17");
  assert.match(await page.locator("#team-metrics").textContent(), /Детали скрыты/);
  assert.equal(await page.locator("#team-metrics dd").count(), 0);
  if (output) await page.screenshot({ path: path.join(output, "divisions-" + viewport.width + ".png") });
  await page.locator("[data-view='overview']").click();

  await page.locator("#previous-signal [data-team-id='T16']").click();
  assert.equal(await page.locator("#wave").inputValue(), "2026-06");
  assert.equal(await page.locator("#team").inputValue(), "T16");
  assert.match(await page.locator("#detail-signals").textContent(), /После поправки.*q 0,018/);
  await page.locator("[data-view='overview']").click();
  await page.locator("#wave").selectOption("2026-06");
  assert.equal(await page.locator("#trend circle").count(), 2);
  assert.deepEqual(await page.locator("#signals .signal.confirmed strong").allTextContents(), ["T16"]);
  assert.deepEqual(await page.locator("#market-rows td").allTextContents(), ["14,42%", "13,12%", "21,1%", "18,6%", "27%", "19%"]);
  await page.locator("#wave").selectOption("2026-03");
  assert.equal(await page.locator("#trend circle").count(), 1);
  assert.match(await page.locator("#signals").textContent(), /предшествующего сравнения нет/);
  await page.locator("#wave").selectOption("2026-09");

  await page.locator("#retail-focus button").click();
  assert.equal(await page.locator("#division-rows tr").count(), 5);
  assert.equal(await page.locator("#division").inputValue(), "Розничный блок");
  await page.getByRole("button", { name: "Розничный блок" }).click();
  assert.equal(await page.locator("#division").inputValue(), "");
  await page.getByRole("button", { name: "Розничный блок" }).click();
  assert.equal(await page.locator("#division").inputValue(), "Розничный блок");
  assert.equal(await page.locator("#annual-value").textContent(), "20,19%");
  assert.equal(await page.locator("#division-rows tr.selected").count(), 1);
  await page.locator("[data-view='comments']").click();
  assert.match(await page.locator("#comments-scope").textContent(), /Розничный блок/);
  assert.match(await page.locator("#topic-score-scope").textContent(), /По всем дивизионам выбранной волны/);
  assert.ok(await page.locator("#score-links div").count() > 0);
  const scoped = await page.locator("#comments-scope").textContent();
  await page.locator("#division").selectOption("");
  assert.notEqual(await page.locator("#comments-scope").textContent(), scoped);
  assert.match(await page.locator("#comments-scope").textContent(), /379/);
  assert.match(await page.locator("#topics .negative").textContent(), /Нагрузка11,3/);
  const caveat = await page.locator("#comments .evidence-note").boundingBox();
  assert.ok(caveat && caveat.y + caveat.height <= viewport.height, "model caveat extends past the viewport");
  if (output) await page.screenshot({ path: path.join(output, "comments-" + viewport.width + ".png") });

  await page.locator("#open-method").click();
  assert.equal(await page.locator("dialog").isVisible(), true);
  assert.match(await page.locator("#method-model").textContent(), /Отношение шансов.*0,82.*95% ДИ/);
  await page.keyboard.press("Escape");
  assert.equal(await page.locator("dialog").isVisible(), false);

  const source = await page.locator("#snapshot").textContent();
  assert.ok(!/employee_id|comment_id|text_id|EMP\d+/i.test(source));
  assert.deepEqual(pageErrors, []);
  assert.deepEqual(externalRequests, []);
  await context.close();
  console.log(viewport.width + "×" + viewport.height + " offline browser checks passed");
}

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH, headless: true });
  try {
    for (const viewport of [{ width: 1366, height: 768 }, { width: 1920, height: 1080 }]) {
      await verify(browser, viewport);
    }
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
