# Дашборд на GitHub Pages

[Готовая панель](../dashboard/index.html) открывается из локальной папки в браузере. HTML содержит оформление, код и агрегированные данные. Файлы `dashboard/app.js`, `dashboard/style.css` и `dashboard/template.html` нужны для сборки HTML.

## Ручной запуск после согласования

1. В репозитории GitHub откройте **Settings → Pages** и выберите **GitHub Actions** как источник публикации.
2. В **Actions → Publish HR dashboard** нажмите **Run workflow** и выберите основную ветку.
3. После успешного завершения откройте адрес панели из шага **Deploy to GitHub Pages** или раздела **Settings → Pages**.

Сценарий [dashboard-pages.yml](../.github/workflows/dashboard-pages.yml) запускается вручную. В артефакт сайта копируется только `dashboard/index.html` как `index.html`. Проверьте доступ к сайту отдельно от видимости репозитория. [Документация GitHub Pages о видимости сайта](https://docs.github.com/en/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site).

Обновленную версию панели публикуйте тем же сценарием после проверки и загрузки новой сборки в основную ветку. [Документация GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
