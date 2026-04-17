# 🔄 ПОСЛЕ ПЕРЕЗАГРУЗКИ — читать первым делом

## Контекст (почему перезагрузка)

Зависание было из-за того что Python 3.11 из `/usr/local/bin/` — **x86_64 сборка**,
запускалась через Rosetta 2 на Apple Silicon (M1/M2/M3).
PyTorch пытался скомпилировать 600MB библиотеку через эмулятор → зависание.

---

## Шаг 1 — Установить нативный ARM Python через pyenv

```bash
# Установить pyenv если нет
brew install pyenv

# Добавить pyenv в PATH (если не добавлен — проверь ~/.zshrc)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# Установить нативный ARM Python 3.11
pyenv install 3.11.9

# Проверить что это ARM (должно быть arm64)
pyenv local 3.11.9
python --version
python -c "import platform; print(platform.machine())"  # должно быть: arm64
```

## Шаг 2 — Пересоздать виртуальное окружение

```bash
cd /Users/pro/git/DINOv2

# Удалить старое окружение (оно было x86)
rm -rf .venv

# Создать новое на ARM Python
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate

# Проверить архитектуру
python -c "import platform; print(platform.machine())"  # arm64

# Установить зависимости
pip install -r requirements.txt
```

## Шаг 3 — Проверить что torch видит MPS (Apple GPU)

```bash
source .venv/bin/activate
python -c "
import torch
print('PyTorch:', torch.__version__)
print('MPS доступен:', torch.backends.mps.is_available())
# Должно быть True — это Apple Silicon GPU
"
```

## Шаг 4 — Включить MPS в extractor.py

После установки ARM Python — MPS будет работать без зависаний.
Нужно раскомментировать MPS в `app/extractor.py`:

Найти функцию `get_device()` и заменить:
```python
# БЫЛО (cpu-only):
return torch.device("cpu")

# СТАЛО (MPS для Apple Silicon):
if torch.backends.mps.is_available():
    return torch.device("mps")
return torch.device("cpu")
```

Или просто написать мне "готово после перезагрузки" — я сам всё поправлю.

## Шаг 5 — Запустить индексацию (ОДИН процесс!)

```bash
cd /Users/pro/git/DINOv2
source .venv/bin/activate

# Удалить незавершённые индексы от предыдущих попыток
rm -f index/faiss.index index/embeddings.npy index/metadata.json

# Запустить индексацию — ОДИН РАЗ, не запускать повторно пока не завершится!
python build_index.py
```

На MPS индексация займёт ~2-3 минуты вместо 9 минут на CPU.

## Шаг 6 — Запустить веб-приложение

```bash
cd /Users/pro/git/DINOv2
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Открыть в браузере: **http://localhost:8000**

---

## Что уже сделано и работает ✅

- `app/extractor.py` — извлечение 1536-dim эмбеддингов через DINOv2
- `app/indexer.py` — построение FAISS индекса из 260 пустых комнат
- `app/retriever.py` — поиск топ-5 похожих комнат
- `app/config.py` — пути и константы
- `main.py` — FastAPI сервер (6 endpoints)
- `static/index.html` + `style.css` + `app.js` — веб-интерфейс (drag & drop, dark mode)
- `build_index.py` — скрипт запуска индексации
- `requirements.txt` — зависимости (без transformers!)
- `database/` — 260 пустых + 260 с мебелью (520 картинок)

## Единственное что осталось

Запустить индексацию (шаги 1-6 выше) и протестировать поиск.
