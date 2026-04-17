# RoomFinder — Поиск похожих пустых комнат

Система поиска похожих комнат по геометрии, ракурсу и пространственному расположению объектов. Загружаешь фото пустой комнаты — система находит максимально похожие варианты в базе из ~600 комнат.

---

## Архитектура

```
database/
  before/   ← индексируемые фото (пустые комнаты)
  after/    ← итоговые фото (меблированные, для превью результата)
index/
  faiss.index         ← DINOv2 индекс (1536-dim)
  depth.index         ← Depth индекс (4608-dim)
  metadata.json       ← маппинг ID → имя файла
  depth_metadata.json ← маппинг для depth индекса
  yolo_metadata.json  ← сектора окон/дверей (3×3 сетка)
app/
  config.py           ← константы (пути, размеры, TOP_K)
  extractor.py        ← DINOv2 эмбеддинги
  depth_extractor.py  ← Depth Anything V2 эмбеддинги
  yolo_extractor.py   ← YOLOv8 детекция и пространственная сетка
  indexer.py          ← построение FAISS индекса
  retriever.py        ← поиск (3 режима) + YOLO reranking
main.py               ← FastAPI API + serve статики
static/               ← фронтенд (HTML, CSS, JS)
```

---

## Стек технологий

### ML модели
| Модель | Назначение | Размер | HF / Hub |
|---|---|---|---|
| `facebook/dinov2-vitb14` | Визуальные эмбеддинги (текстура, ракурс, стиль) | ~330 MB | `torch.hub` |
| `depth-anything/Depth-Anything-V2-Small-hf` | Карты глубины → 3D геометрия | ~100 MB | HuggingFace |
| `yolov8m-oiv7.pt` | Детекция окон и дверей (расположение в сетке 3×3) | ~50 MB | Ultralytics Hub |

### Backend
| Пакет | Версия | Назначение |
|---|---|---|
| `torch` | 2.2.2 | Инференс DINOv2 (MPS / CPU) |
| `torchvision` | 0.17.2 | Препроцессинг изображений |
| `faiss-cpu` | ≥1.7.4 | Approximate Nearest Neighbor поиск |
| `transformers` | ≥4.40, <5.0 | Загрузка Depth Anything V2 |
| `huggingface-hub` | ≥0.20 | Скачивание моделей с HF |
| `ultralytics` | latest | YOLOv8 inference |
| `fastapi` | ≥0.110 | REST API |
| `uvicorn[standard]` | ≥0.29 | ASGI сервер |
| `python-multipart` | ≥0.0.9 | Загрузка файлов через форму |
| `Pillow` | ≥10.0 | Работа с изображениями |
| `numpy` | `==1.26.4` | Векторные операции (pinned, совместимость) |
| `scipy` | ≥1.11 | Sobel градиент для depth эмбеддингов |
| `opencv-python-headless` | ≥4.8 | GaussianBlur для depth карты |
| `tqdm` | ≥4.66 | Прогресс-бары при индексации |

---

## Принцип работы

### 1. Индексация базы (один раз)
Запускается при первом старте или через build-скрипты:

```bash
# DINOv2 индекс (обязательный)
python build_index.py

# Depth Anything V2 индекс (для гибридного и depth-only режимов)
python build_depth_index.py

# YOLO пространственный индекс (для spatial reranking)
python build_yolo_index.py
```

### 2. DINOv2 эмбеддинги
Каждая фотография прогоняется через `dinov2_vitb14` (ViT-B/14, patch_size=14, image_size=518).

**Формула:**
```
embedding = L2_norm( concat(CLS_token, mean(patch_tokens)) )
             [768]           +       [768]              → [1536 dim]
```

Вектор L2-нормализован → FAISS использует Inner Product как косинусное сходство.

### 3. Depth эмбеддинги
`Depth-Anything-V2-Small` строит монокулярную карту глубины. Из неё извлекается:
- **Raw depth 48×48** — общая 3D структура (закрытая/открытая комната, высота потолка)
- **Sobel gradient 48×48** — границы поверхностей (углы стен, рамы окон, где пол переходит в стену)

```
depth_embedding = L2_norm( concat(raw_depth_flat, sobel_grad_flat) )
                                 [2304]    +      [2304]         → [4608 dim]
```

### 4. YOLO пространственная сетка
`YOLOv8m-oiv7` детектирует окна (class_id=587) и двери (class_id=164). Кадр делится на сетку **3×3** (9 секторов: `top-left`, `mid-center`, `bottom-right`, и т.д.). Для каждого объекта записывается сектор его центра.

```json
{
  "bc_001_before.jpg": {
    "windows": ["top-left", "top-center"],
    "doors": ["bottom-right"]
  }
}
```

---

## Режимы поиска

### Режим 1: DINOv2 Only
FAISS Inner Product поиск по 1536-dim эмбеддингам.

```
score = cosine_similarity(query_dino_vec, candidate_dino_vec)
```

### Режим 2: Hybrid (DINOv2 + Depth)
DINOv2 отбирает **30 кандидатов**. Для них вычисляется depth score и результаты объединяются:

```
combined_score = DINO_score × 0.35 + Depth_score × 0.65
```
> Depth получает больший вес, т.к. несёт чистую геометрическую информацию без влияния цвета/текстур стен.

### Режим 3: Depth Only (Экспериментальный)
Прямой FAISS поиск по 4608-dim depth векторам. DINOv2 полностью игнорируется.

```
score = cosine_similarity(query_depth_vec, candidate_depth_vec)
```
> Режим ищет комнаты с одинаковой геометрией и ракурсом вне зависимости от цвета стен, паркета, освещения.

### Пост-фильтр: YOLO Spatial Reranking
Применяется поверх любого базового режима. Использует **мультипликативную** корректировку:

```python
spatial_multiplier = 1.0
# За каждое совпавшее окно/дверь в том же секторе:
spatial_multiplier += 0.10
# За каждое окно/дверь в другом секторе (или не найденное):
spatial_multiplier -= 0.05

final_score = clip(base_score × spatial_multiplier, 0.0, 1.0)
```

> Мультипликативная схема гарантирует, что YOLO не может "протолкнуть" изначально слабого кандидата в топ — сильный скор делает хорошей комнате больший бонус, чем слабой.

---

## Фильтрация результатов

### Жёсткий порог (Hard Cutoff)
Только результаты с `score >= 88%` считаются точными совпадениями.

### Fallback (Мягкое падение)
Если после жёсткого отсева список пуст — система снижает порог до `80%` и возвращает топ-3 с флагом `is_fallback: true`. На фронтенде отображается уведомление *"Точных совпадений нет, показываем наиболее похожие варианты"*.

---

## API Reference

### `POST /search`
Основной эндпоинт поиска.

**Form Data параметры:**
| Параметр | Тип | Default | Описание |
|---|---|---|---|
| `file` | `UploadFile` | — | Фото комнаты (JPG, PNG) |
| `use_dinov2` | `bool` | `true` | Включить DINOv2 в поиск |
| `use_depth` | `bool` | `false` | Включить Depth Anything V2 |
| `use_yolo` | `bool` | `false` | Включить YOLO spatial reranking |

> Если `use_dinov2=false` и `use_depth=false` → HTTP 400 Bad Request.

**Ответ:**
```json
{
  "query_filename": "uuid.jpg",
  "mode": "hybrid + YOLO",
  "is_fallback": false,
  "results": [
    {
      "rank": 1,
      "filename": "bc_012_before.jpg",
      "after_filename": "after/bc_012_after.jpg",
      "score": 0.9432,
      "score_pct": 94.3,
      "mode": "hybrid + YOLO",
      "score_dino": 91.2,
      "score_depth": 96.1,
      "yolo_bonus": 1.1
    }
  ]
}
```

### `POST /visualize`
Генерирует debug-визуализации для уже загруженного изображения.

**Form Data:** `filename` (имя файла из `/uploads/`)

**Ответ:**
```json
{
  "depth_img": "<base64 PNG>",
  "yolo_img": "<base64 PNG>"
}
```

### `POST /build-index`
Перестроить DINOv2 индекс из базы.

### `GET /status`
Статус индексов (построен/не построен, количество векторов).

---

## Запуск

### Требования
- Python 3.11+
- macOS (Apple Silicon MPS) или Linux (CPU)
- ~4 GB RAM (все три модели в памяти одновременно)
- [`uv`](https://docs.astral.sh/uv/) — менеджер пакетов

### Установка

```bash
# 1. Установить uv (один раз глобально)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Клонировать репозиторий
git clone git@github.com:AndreiSklaruk/DinoV2-YOLO-Depth_find_same_interiour_service.git
cd DinoV2-YOLO-Depth_find_same_interiour_service

# 3. Создать окружение и установить все зависимости (одна команда)
uv sync
```

> `uv sync` читает `pyproject.toml` и `uv.lock`, создаёт `.venv` и устанавливает точные версии пакетов. Работает за ~30 секунд даже на холодной машине.

### Индексация базы
```bash
# 1. Разложить фото в database/before/ (формат: bc_NNN_before.jpg)
# 2. Разложить итоговые фото в database/after/ (формат: bc_NNN_after.jpg)

python build_index.py        # ~5-15 мин, создаёт index/faiss.index
python build_depth_index.py  # ~10-20 мин, создаёт index/depth.index
python build_yolo_index.py   # ~2-5 мин, создаёт index/yolo_metadata.json
```

### Запуск сервера
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Открыть в браузере: **http://localhost:8000**

---

## Структура результата в UI

Карточки отображаются в двух ракурсах:
- **Before** — пустая комната из базы (что нашла система)
- **After** — меблированный вариант этой же комнаты (из `database/after/`)

Вкладки визуализации для запроса:
- **Оригинал** — uploaded фото
- **Depth Map** — карта глубины (Turbo colormap, синий = близко, красный = далеко)  
- **YOLO Objects** — детекция окон/дверей с сеткой 3×3

---

## Конфигурация (`app/config.py`)

| Параметр | Значение | Описание |
|---|---|---|
| `MODEL_NAME` | `facebook/dinov2-base` | Имя модели (torch.hub) |
| `IMAGE_SIZE` | `518` | Входной размер DINOv2 (кратен patch_size=14) |
| `EMBEDDING_DIM` | `1536` | Размерность DINOv2 эмбеддинга |
| `TOP_K` | `10` | Максимум результатов |
| `DEPTH_MAP_SIZE` | `48` | Разрешение depth карты при векторизации (48×48) |
| `DEPTH_EMBEDDING_DIM` | `4608` | Размерность depth эмбеддинга (48×48×2) |

---

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `PYTORCH_ENABLE_MPS_FALLBACK` | `0` | `1` — разрешить CPU fallback для несупортируемых MPS операций |
| `DINOV2_FORCE_CPU` | `0` | `1` — принудительно использовать CPU вместо MPS |
