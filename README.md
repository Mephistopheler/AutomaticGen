# AQG: русскоязычная адаптация (траектория A)

Готовый минимальный проект для дипломной работы по автоматической генерации вопросов.

В проекте есть:
- обучение модели `context + answer -> question`;
- конфиги для `google/mt5-base` и `ai-forever/ruT5-base`;
- загрузка `SQuAD`, `SberQuAD` и локальных JSONL/CSV-файлов;
- оценка по SacreBLEU, ROUGE, METEOR, BERTScore;
- генерация вопросов на своих примерах;
- подготовка CSV для human evaluation.

---

## 1. Что именно здесь реализовано

Этот код закрывает большую часть практической части диплома по траектории A:

1. **Базовый англоязычный этап** — обучение на SQuAD.
2. **Русскоязычная адаптация** — дообучение и оценка на SberQuAD.
3. **Сравнение baselines** — `mT5` против `ruT5`.
4. **Оценка качества** — автоматические метрики + заготовка для human evaluation.
5. **Доменная адаптация** — шаблон конфига для своих учебных текстов.

---

## 2. Структура проекта

```text
aqg_ru_adaptation/
├─ README.md
├─ requirements.txt
├─ train.py
├─ evaluate.py
├─ generate.py
├─ create_human_eval_sheet.py
├─ configs/
│  ├─ mt5_en_to_ru_transfer.yaml
│  ├─ mt5_sberquad_finetune.yaml
│  ├─ rut5_sberquad_finetune.yaml
│  └─ domain_adaptation_template.yaml
├─ data/
│  └─ examples/
│     ├─ domain_train.jsonl
│     └─ generate_input.jsonl
└─ src/
   └─ aqg/
      ├─ config_utils.py
      ├─ data.py
      ├─ metrics.py
      └─ utils.py
```

---

## 3. Что нужно установить

### Вариант для Windows PowerShell

Открой PowerShell в папке проекта и выполни:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если у тебя несколько версий Python, используй `py -3.10` или `py -3.11` вместо `python`.

### Проверка

```powershell
python -c "import torch, transformers, datasets; print('ok')"
```

---

## 4. Какой эксперимент запускать первым

### Эксперимент 1. mT5: EN -> RU перенос

Это сценарий для раздела про transfer learning.

```powershell
python train.py --config configs/mt5_en_to_ru_transfer.yaml
python evaluate.py --config configs/mt5_en_to_ru_transfer.yaml --checkpoint outputs/mt5_en_to_ru_transfer/checkpoints/best
```

Что получится:
- обучение на SQuAD;
- проверка на русских примерах из SberQuAD;
- метрики для блока zero-shot / cross-lingual transfer.

### Эксперимент 2. mT5: fine-tuning на SberQuAD

```powershell
python train.py --config configs/mt5_sberquad_finetune.yaml
python evaluate.py --config configs/mt5_sberquad_finetune.yaml --checkpoint outputs/mt5_sberquad_finetune/checkpoints/best
```

### Эксперимент 3. ruT5: fine-tuning на SberQuAD

```powershell
python train.py --config configs/rut5_sberquad_finetune.yaml
python evaluate.py --config configs/rut5_sberquad_finetune.yaml --checkpoint outputs/rut5_sberquad_finetune/checkpoints/best
```

Это и будет главным сравнением для диплома:
- `mT5 + SberQuAD`
- `ruT5 + SberQuAD`

---

## 5. Где смотреть результаты

После запуска появляются папки вида:

```text
outputs/mt5_sberquad_finetune/
├─ checkpoints/
│  ├─ epoch_1/
│  ├─ epoch_2/
│  └─ best/
├─ logs/
│  ├─ training_summary.json
│  ├─ validation_metrics.json
│  └─ validation_predictions.jsonl
└─ evaluation/
   ├─ metrics.json
   └─ predictions.jsonl
```

### Что брать в диплом

Из этих файлов тебе нужны:

- `training_summary.json` — для описания обучения;
- `validation_metrics.json` — для таблицы валидации;
- `evaluation/metrics.json` — для итоговой таблицы результатов;
- `evaluation/predictions.jsonl` — для качественного анализа ошибок и human evaluation.

---

## 6. Как вставить результаты в диплом

Когда получишь числа, вставляй их в такие блоки.

### Таблица 1. Сравнение моделей на SberQuAD

| Модель | Обучение | SacreBLEU | ROUGE-L | METEOR | BERTScore F1 |
|---|---|---:|---:|---:|---:|
| mT5-base | SQuAD -> SberQuAD eval | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] |
| mT5-base | SberQuAD fine-tuning | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] |
| ruT5-base | SberQuAD fine-tuning | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] |

### Таблица 2. Human evaluation

| Модель | Беглость | Релевантность | Answerability | Педагогическая ценность | Итог |
|---|---:|---:|---:|---:|---:|
| mT5-base | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] |
| ruT5-base | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] | [ТВОЁ ЧИСЛО] |

---

## 7. Как сгенерировать вопросы на своих примерах

### Один пример

```powershell
python generate.py --config configs/rut5_sberquad_finetune.yaml --checkpoint outputs/rut5_sberquad_finetune/checkpoints/best --context "Фотосинтез — это процесс, при котором растения преобразуют световую энергию в химическую." --answer "процесс, при котором растения преобразуют световую энергию в химическую"
```

### Несколько примеров из JSONL

```powershell
python generate.py --config configs/rut5_sberquad_finetune.yaml --checkpoint outputs/rut5_sberquad_finetune/checkpoints/best --input_jsonl data/examples/generate_input.jsonl
```

Формат входного JSONL:

```json
{"context": "...", "answer_text": "..."}
```

---

## 8. Как провести human evaluation

После `evaluate.py` у тебя будет файл:

```text
outputs/.../evaluation/predictions.jsonl
```

Преобразуй его в таблицу для аннотаторов:

```powershell
python create_human_eval_sheet.py --predictions outputs/rut5_sberquad_finetune/evaluation/predictions.jsonl --output outputs/rut5_sberquad_finetune/evaluation/human_eval_sheet.csv
```

Получится CSV, где уже есть колонки:
- `fluency_score_1_5`
- `relevance_score_1_5`
- `answerability_score_1_5`
- `pedagogical_value_score_1_5`
- `overall_score_1_5`
- `annotator_id`
- `notes`

Дальше можно отправить этот CSV двум-трём экспертам.

---

## 9. Как использовать свои учебные тексты

Если ты хочешь сделать доменную адаптацию, используй конфиг:

```text
configs/domain_adaptation_template.yaml
```

Сейчас он смотрит на примерный файл:

```text
data/examples/domain_train.jsonl
```

### Нужный формат локального датасета

Поддерживается SQuAD-подобный JSONL:

```json
{"id":"1","context":"...","question":"...","answers":{"text":["..."],"answer_start":[0]}}
```

Ты можешь просто заменить содержимое `domain_train.jsonl` на свои данные.

Запуск:

```powershell
python train.py --config configs/domain_adaptation_template.yaml
python evaluate.py --config configs/domain_adaptation_template.yaml --checkpoint outputs/domain_adaptation_template/checkpoints/best
```

---

## 10. Что писать в разделе «Программная реализация»

Можно использовать примерно такую логику описания:

1. Реализован модуль загрузки и нормализации датасетов.
2. Все корпуса приводятся к единому формату: `context`, `answer_text`, `question`.
3. Вход модели формируется как строка вида:

```text
generate question: context: {context} answer: {answer}
```

4. Для обучения применяются encoder-decoder модели семейства T5.
5. Поддержаны два baseline-сценария: `google/mt5-base` и `ai-forever/ruT5-base`.
6. Обучение выполняется в режиме fine-tuning на задаче генерации вопроса по паре `(контекст, ответ)`.
7. Оценивание реализовано через SacreBLEU, ROUGE, METEOR и BERTScore.
8. Для качественного анализа формируется JSONL-файл с примерами предсказаний.
9. Для экспертной оценки автоматически формируется CSV-анкета.

---

## 11. Что делать, если памяти не хватает

Для твоего ноутбука это важно.

Если обучение падает по памяти:

1. уменьши `batch_size` до `1` или `2`;
2. увеличь `gradient_accumulation_steps`;
3. уменьшай `max_source_length` с `384` до `256`;
4. начни с `mt5-small` или `rut5-base`, если `base` работает тяжело;
5. сначала проверь всё на маленьком локальном датасете.

---

## 12. Самый короткий путь до первых результатов

Вот минимальная последовательность команд:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --config configs/rut5_sberquad_finetune.yaml
python evaluate.py --config configs/rut5_sberquad_finetune.yaml --checkpoint outputs/rut5_sberquad_finetune/checkpoints/best
python create_human_eval_sheet.py --predictions outputs/rut5_sberquad_finetune/evaluation/predictions.jsonl --output outputs/rut5_sberquad_finetune/evaluation/human_eval_sheet.csv
```

После этого у тебя уже будут:
- чекпоинт модели;
- итоговые метрики;
- примеры генерации;
- CSV для экспертной оценки.

---

## 13. Что тебе нужно будет дописать вручную в диплом

Только то, что невозможно честно сгенерировать без реального запуска:

- реальные размеры итоговых train/validation/test выборок;
- фактические гиперпараметры финального лучшего запуска;
- итоговые значения метрик;
- примеры лучших и худших вопросов;
- результаты human evaluation;
- интерпретацию, почему `mT5` или `ruT5` оказалась лучше.

---

## 14. Быстрая проверка, что всё работает

Если хочешь сначала не гонять большие датасеты, протестируй на локальном примере:

```powershell
python train.py --config configs/domain_adaptation_template.yaml
python generate.py --config configs/domain_adaptation_template.yaml --checkpoint outputs/domain_adaptation_template/checkpoints/best --input_jsonl data/examples/generate_input.jsonl
```

Это полезно, чтобы убедиться, что пайплайн запускается от начала до конца.
