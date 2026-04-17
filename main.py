"""
main.py — FastAPI приложение для поиска похожих пустых комнат.

Запуск:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import base64
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.config import DATABASE_DIR, STATIC_DIR, UPLOADS_DIR, TOP_K
from app.extractor import load_model
from app.indexer import build_index, is_index_built, load_index
from app.retriever import search, search_hybrid, search_depth_only, apply_yolo_rerank, apply_mirror_penalty
import json

# ─── Глобальное состояние ─────────────────────────────────────────────────────
_faiss_index = None
_metadata: dict[int, str] | None = None
_index_count: int = 0
_depth_index = None
_yolo_metadata = None

app = FastAPI(
    title="RoomFinder",
    description="Поиск похожих пустых комнат по геометрии, ракурсу и структуре",
    version="3.0.0",
)


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global _faiss_index, _metadata, _index_count

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    load_model()

    if not is_index_built():
        print("[Startup] Индекс не найден, запускаю построение...")
        build_index()

    _faiss_index, _metadata = load_index()
    _index_count = _faiss_index.ntotal
    print(f"[Startup] ✅ Готово. DINOv2 индекс: {_index_count} комнат.")


def _load_yolo_metadata():
    global _yolo_metadata
    if _yolo_metadata is not None:
        return _yolo_metadata
    yolo_path = Path("index/yolo_metadata.json")
    if not yolo_path.exists():
        return None
    with open(yolo_path, "r", encoding="utf-8") as f:
        _yolo_metadata = json.load(f)
    print(f"[YOLO] Метаданные загружены")
    return _yolo_metadata


def _load_depth_index():
    """Lazy-загрузка Depth индекса."""
    global _depth_index
    if _depth_index is not None:
        return _depth_index
    depth_path = Path("index/depth_faiss.index")
    if not depth_path.exists():
        return None
    import faiss
    _depth_index = faiss.read_index(str(depth_path))
    print(f"[Depth] Индекс загружен: {_depth_index.ntotal} векторов")
    return _depth_index



# ─── Static files ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/status")
async def status():
    """Статус сервера и индексов."""
    depth_ready = Path("index/depth_faiss.index").exists()
    yolo_ready = Path("index/yolo_metadata.json").exists()
    return {
        "status": "ok",
        "indexed_rooms": _index_count,
        "model": "facebook/dinov2-base",
        "index_ready": _faiss_index is not None,
        "depth_index_ready": depth_ready,
        "yolo_index_ready": yolo_ready,
    }


@app.get("/images/{filepath:path}")
async def get_database_image(filepath: str):
    img_path = DATABASE_DIR / filepath
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Изображение не найдено: {filepath}")
    return FileResponse(str(img_path), media_type="image/jpeg")


@app.get("/uploads/{filename}")
async def get_upload_image(filename: str):
    upload_path = UPLOADS_DIR / filename
    if not upload_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(str(upload_path), media_type="image/jpeg")


@app.post("/search")
async def search_similar(
    file: UploadFile = File(...),
    use_dinov2: bool = Form(True),
    use_depth: bool = Form(False),
    use_yolo: bool = Form(False),
    use_mirror: bool = Form(False),
):
    """
    Поиск похожих комнат.

    Params:
        use_depth: включить Depth Anything V2

    Режимы:
        False  → DINOv2 only
        True   → DINOv2 + Depth (15/85%)
    """
    if _faiss_index is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Индекс не загружен.")

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат: {file.content_type}")

    contents = await file.read()
    try:
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось открыть изображение: {e}")

    upload_name = f"{uuid.uuid4().hex}.jpg"
    upload_path = UPLOADS_DIR / upload_name
    query_image.save(str(upload_path), "JPEG", quality=90)

    try:
        # Выбор режима по комбинации флагов
        if not use_dinov2 and not use_depth:
            raise HTTPException(status_code=400, detail="Выберите хотя бы один метод поиска (DINOv2 или Depth).")

        if use_dinov2 and use_depth:
            # Hybrid: DINOv2 + Depth
            depth_idx = _load_depth_index()
            if depth_idx is None:
                raise HTTPException(status_code=503, detail="Depth индекс не найден. Запустите: python build_depth_index.py")
            results = search_hybrid(
                query_image=query_image,
                dino_index=_faiss_index,
                depth_index=depth_idx,
                metadata=_metadata,
                top_k=20,  # Запрашиваем больше для последующей фильтрации
            )
            mode = "hybrid"

        elif use_dinov2 and not use_depth:
            # DINOv2 only
            results = search(
                query_image=query_image,
                index=_faiss_index,
                metadata=_metadata,
                top_k=20,  # Запрашиваем больше для последующей фильтрации
            )
            mode = "dinov2"

        elif not use_dinov2 and use_depth:
            # Depth only
            depth_idx = _load_depth_index()
            if depth_idx is None:
                raise HTTPException(status_code=503, detail="Depth индекс не найден.")
            results = search_depth_only(
                query_image=query_image,
                depth_index=depth_idx,
                metadata=_metadata,
                top_k=20,
            )
            mode = "depth-only"

        if use_yolo:
            yolo_meta = _load_yolo_metadata()
            if yolo_meta is not None:
                try:
                    from app.yolo_extractor import extract_yolo_grid
                    query_yolo = extract_yolo_grid(query_image)
                    results = apply_yolo_rerank(results, query_yolo, yolo_meta)
                except Exception as e:
                    print(f"[YOLO Error] {e}")

        if use_mirror:
            depth_idx = _load_depth_index()
            if depth_idx is not None:
                try:
                    results = apply_mirror_penalty(
                        results=results,
                        query_image=query_image,
                        depth_index=depth_idx,
                        metadata=_metadata,
                    )
                except Exception as e:
                    print(f"[Mirror Error] {e}")

        # Фильтрация только > 88%
        filtered_results = [r for r in results if r.get("score_pct", 0) >= 88.0]
        is_fallback = False
        
        if len(filtered_results) == 0:
            # Fallback к 80%, берем топ-3
            filtered_results = [r for r in results if r.get("score_pct", 0) >= 80.0][:3]
            is_fallback = True
        
        # Берем только TOP_K (обычно 4)
        if not is_fallback:
            final_results = filtered_results[:TOP_K]
        else:
            final_results = filtered_results
        
        # Пересчет рангов
        for i, r in enumerate(final_results, start=1):
            r["rank"] = i
            
        results = final_results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {e}")

    return JSONResponse({
        "query_filename": upload_name,
        "mode": mode,
        "is_fallback": is_fallback,
        "results": final_results,
    })


@app.post("/visualize")
async def visualize_analysis(filename: str = Form(...)):
    """
    Сгенерировать debug-визуализации для уже загруженного изображения.

    Params:
        filename: имя файла в uploads/ (возвращается полем query_filename из /search)

    Returns:
        {
          "depth_img": "data:image/png;base64,...",
        }
    """
    upload_path = UPLOADS_DIR / filename
    if not upload_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")

    try:
        image = Image.open(upload_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось открыть: {e}")

    def pil_to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    try:
        from app.depth_extractor import get_depth_visualization
        from app.yolo_extractor import get_yolo_visualization

        # Depth visualization (turbo colormap)
        depth_pil = get_depth_visualization(image)
        depth_b64 = pil_to_b64(depth_pil)
        
        # YOLO visualization
        yolo_pil = get_yolo_visualization(image)
        yolo_b64 = pil_to_b64(yolo_pil)

        return JSONResponse({
            "depth_img": depth_b64,
            "yolo_img": yolo_b64,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка визуализации: {e}")



@app.post("/rebuild-index")

async def rebuild_index():
    """Пересобрать DINOv2 индекс."""
    global _faiss_index, _metadata, _index_count
    try:
        stats = build_index()
        _faiss_index, _metadata = load_index()
        _index_count = _faiss_index.ntotal
        return {"status": "ok", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
