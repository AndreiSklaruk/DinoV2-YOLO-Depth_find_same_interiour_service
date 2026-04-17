# RoomFinder — Документация проекта
**Последнее обновление:** 2026-04-08  
**Python:** 3.11.9  
**Платформа:** macOS Apple Silicon (MPS backend)

---

## Что это

FastAPI веб-приложение для поиска похожих пустых комнат по геометрии, структуре и ракурсу.  
Пользователь загружает фото пустой комнаты → система находит top-4 наиболее похожих из базы 260 изображений.

**Главная задача:** Находить комнаты с максимально похожей геометрией (коробка комнаты, ракурс, пропорции).  
Это нужно для переноса маски мебели с референсного фото на новое — пространственное совпадение критично.

---

## Архитектура (текущее состояние)

```
/Users/pro/git/DINOv2/
├── main.py                    # FastAPI приложение (входная точка)
├── build_index.py             # Построение DINOv2 FAISS-индекса
├── build_depth_index.py       # Построение Depth Anything V2 FAISS-индекса

├── requirements.txt           # Python зависимости
├── .python-version            # Python 3.11.9
├── AFTER_REBOOT.md            # Инструкции после перезагрузки
├── PROJECT_STATE.md           # Этот файл
│
├── app/
│   ├── config.py              # Конфигурация (пути, размеры, TOP_K=4)
│   ├── extractor.py           # DINOv2 эмбеддинг
│   ├── depth_extractor.py     # Depth Anything V2 эмбеддинг

│   ├── indexer.py             # Построение и загрузка FAISS-индекса
│   └── retriever.py           # Логика поиска (4 режима)
│
├── static/
│   ├── index.html             # Главная страница (двухпанельный UI)
│   ├── app.js                 # Клиентская логика
│   └── style.css              # Стили (dark theme, glassmorphism)
│
├── database/                  # 252 фото пустых комнат + _after с мебелью
│   ├── image_1.jpg            # Все: 1280×852px (соотношение 3:2)
│   ├── image_1_after.jpg
│   └── ...
│
├── index/                     # FAISS-индексы (бинарные файлы)
│   ├── faiss.index            # DINOv2 индекс (252 × 1536 dim)
│   ├── metadata.pkl           # id → filename маппинг
│   ├── depth_faiss.index      # Depth индекс (260 × 4608 dim)
│   ├── depth_embeddings.npy   # Numpy массив depth-эмбеддингов

│
└── uploads/                   # Временные загруженные файлы
```

---

## Установленные библиотеки и версии

| Библиотека | Версия | Назначение |
|---|---|---|
| `torch` | 2.2.2 | PyTorch (MPS backend для Apple Silicon) |
| `torchvision` | 0.17.2 | Компонент PyTorch |
| `transformers` | 4.57.6 | HuggingFace (Depth Anything V2 модель) |
| `huggingface_hub` | ≥0.20.0 | Загрузка моделей с HuggingFace |
| `faiss-cpu` | 1.13.2 | Векторный поиск (IndexFlatIP) |
| `numpy` | 1.26.4 | Работа с массивами эмбеддингов |
| `Pillow` | 12.2.0 | Загрузка и конвертация изображений |
| `scipy` | 1.17.1 | Фильтры Sobel для градиентных признаков |

| `fastapi` | 0.135.3 | REST API сервер |
| `uvicorn` | 0.44.0 | ASGI сервер |
| `python-multipart` | ≥0.0.9 | Парсинг multipart/form-data |
| `tqdm` | ≥4.66.0 | Прогресс-бар при индексации |

> `PYTORCH_ENABLE_MPS_FALLBACK=1` — обязательная переменная окружения для Apple Silicon

---

## Модели и алгоритмы

### 1. DINOv2 (facebook/dinov2-base)
- **Назначение:** Визуальный эмбеддинг изображения
- **Размер вектора:** 1536 dim (CLS 768 + mean patches 768)
- **Загрузка:** Автоматически через `torch.hub` / HuggingFace
- **Роль:** Вспомогательный сигнал (10–15% вес)

### 2. Depth Anything V2 (depth-anything/Depth-Anything-V2-Small-hf)
- **Назначение:** Монокулярная оценка глубины → карта 3D-структуры
- **Размер вектора:** 4608 dim = raw depth (48×48) + Sobel gradient (48×48)
- **Загрузка:** Автоматически через HuggingFace `transformers`
- **Роль:** Главный геометрический сигнал (45–85% вес)

### 3. YOLOv8 (yolov8n-oiv7) [NEW]
- **Назначение:** Дополнительный пространственный фильтр
- **Принцип:** Определяет bounding boxes окон (класс 587) и дверей (класс 164)
- **Grid:** Сетка 3x3. Каждая комната имеет `windows: ["top-left", "center", ...]`
- **Роль:** Дает бонус (+15% к score) если позиции совпали / штраф при конфликте

---

## 4 режима поиска

| Depth | YOLO | Режим | Формула | Mode badge |
|---|---|---|---|---|
| ☐ | ☐ | DINOv2 only | 100% DINOv2 | `DINOv2` |
| ☑ | ☐ | Hybrid | 15% DINOv2 + 85% Depth | `⧆ Depth V2` |
| ☐ | ☑ | DINOv2+YOLO | 100% DINOv2 + бонус совпадений окна/двери | `DINOv2 + YOLO` |
| ☑ | ☑ | Full | Hybrid + бонус совпадений окна/двери | `★ Full (D+Y)` |

**Алгоритм (режимы 2–4):**
1. DINOv2 → top-N кандидатов (N=20–25) из DINOv2-индекса
2. Для каждого кандидата вычисляем score по активным модулям
3. Взвешенная сумма → сортировка → top-4

---

## UI Features

- **Двухпанельный layout:** загруженное фото сверху (50%), результаты снизу (50%)
- **Sidebar (280px):**
  - Зона загрузки (drag & drop / click)
  - Depth Anything V2 toggle + badge (Готов / Недоступен)
  - YOLO Spatial toggle + badge (Готов / Недоступен)
  - Кнопка поиска
  - Статус + счётчик комнат
- **Результаты:** 4 карточки в ряд, aspect-ratio 3:2 (как оригинальные 1280×852)
- **При наведении:** показывает фото "с мебелью" (_after версия)
- **Mode badge:** DINOv2 / ⧆ Depth V2
- **Без скролла:** всё на одном экране

---

## API Endpoints

| Метод | URL | Описание |
|---|---|---|
| `GET` | `/` | Главная страница |
| `GET` | `/status` | Статус индексов и сервера |
| `POST` | `/search` | Поиск похожих комнат |
| `GET` | `/images/{filename}` | Получить фото из базы |
| `POST` | `/rebuild-index` | Пересобрать DINOv2 индекс |

### POST /search параметры
```
file:      image/jpeg или image/png
use_depth: bool (false по умолчанию)
use_yolo:  bool (false по умолчанию)
```

### GET /status ответ
```json
{
  "status": "ok",
  "indexed_rooms": 252,
  "model": "facebook/dinov2-base",
  "index_ready": true,
  "depth_index_ready": true,
  "yolo_index_ready": true
}
```

---

## Как запустить

```bash
cd /Users/pro/git/DINOv2
source .venv/bin/activate

# Обязательно для Apple Silicon:
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Запуск сервера:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Построить DINOv2 индекс (если нет):
python build_index.py

# Построить Depth индекс (если нет):
python build_depth_index.py

```

> Открыть: http://localhost:8000

---

## Git история

| Коммит | Описание |
|---|---|
| `c188fc7` | feat: depth-based floor detection + room calibration |
| `8fd7886` | feat: RoomFinder v2 — DINOv2 + Depth Anything V2 hybrid search |


---


