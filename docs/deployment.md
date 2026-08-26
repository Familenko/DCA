# DCA Web App — Деплоймент

## Що було зроблено (один раз)

### 1. Конфігурація Streamlit

Створено файл `dca/.streamlit/config.toml`:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 8501

[theme]
base = "light"
```

- `headless = true` — Streamlit не відкриває браузер автоматично (потрібно для хмарного хостингу).
- `address = "0.0.0.0"` — слухає всі мережеві інтерфейси, а не тільки `localhost`.
- `port = 8501` — стандартний порт Streamlit.

---

## Деплой на Streamlit Community Cloud (безкоштовно)

1. Запушити репозиторій на GitHub (якщо ще не зроблено):
   ```bash
   git add .
   git commit -m "Add Streamlit cloud config"
   git push
   ```

2. Відкрити [share.streamlit.io](https://share.streamlit.io) і увійти через GitHub.

3. Натиснути **New app**.

4. Вибрати репозиторій і вказати:
   - **Branch**: `prod` (або `main`)
   - **Main file path**: `dca/web_app.py`

5. Натиснути **Deploy**. Streamlit сам встановить залежності з `dca/requirements.txt`.

6. Після деплою отримаєш публічне посилання виду:
   ```
   https://<твій-логін>-<назва-репо>-dcawebapp.streamlit.app
   ```

---

## Локальний запуск

З папки `dca/`:

```bash
streamlit run web_app.py
```

Якщо порт 8501 вже зайнятий — звільнити:

```bash
lsof -ti tcp:8501 | xargs kill -9
```

Потім запустити знову.
