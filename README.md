# Введение в профессию — вариант 18
Павлов Кирилл Кириллович · 15.27Д-ИСТ05/26Б

## Практические работы
- `rest_api_demo.ipynb` — запрос к OMDb API и вывод данных фильма.
- `llm_work.ipynb` — обработка 15 заявок из `rental_18.csv` через LangChain.
- `rental_18_labels.json` — ручная разметка дополнительных полей.
- `reflection1.md`, `reflection2.md` — рефлексии по работам.

## Запуск
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Создайте `.env` по `.env.example` и заполните ключи API. Выберите `.venv` ядром ноутбуков.

Для повторного запуска LLM установите `RUN_API = True` в ноутбуке.
