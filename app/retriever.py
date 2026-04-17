"""
retriever.py — Поиск топ-K похожих пустых комнат через FAISS.

Режимы:
  search()        — только DINOv2
  search_hybrid() — DINOv2 + Depth Anything V2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.config import DATABASE_DIR, TOP_K
from app.extractor import extract_embedding

BEFORE_DIR = DATABASE_DIR / "before"
AFTER_DIR  = DATABASE_DIR / "after"


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _get_after_filename(filename: str) -> str | None:
    """Вернуть относительный URL after-файла для bc_NNN_before.jpg → after/bc_NNN_after.jpg."""
    stem   = Path(filename).stem    # bc_001_before
    suffix = Path(filename).suffix  # .jpg
    if "_before" not in stem:
        return None
    after_stem = stem.replace("_before", "_after")  # bc_001_after
    after_file = AFTER_DIR / (after_stem + suffix)
    if after_file.exists():
        return f"after/{after_stem + suffix}"
    return None


def _make_result(
    rank: int,
    idx: int,
    metadata: dict[int, str],
    score: float,
    mode: str,
    extra: dict | None = None,
) -> dict | None:
    """Собрать словарь результата."""
    filename = metadata.get(idx)
    if filename is None:
        return None
    result = {
        "rank": rank,
        "filename": filename,
        "after_filename": _get_after_filename(filename),
        "score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "score_pct": round(float(np.clip(score, 0.0, 1.0)) * 100, 1),
        "mode": mode,
    }
    if extra:
        result.update(extra)
    return result


# ── Режим 1: DINOv2 only ──────────────────────────────────────────────────────

def search(
    query_image: Image.Image,
    index,
    metadata: dict[int, str],
    top_k: int = TOP_K,
) -> list[dict]:
    """Найти top_k наиболее похожих комнат (только DINOv2)."""
    query_vec = extract_embedding(query_image).reshape(1, -1)
    actual_k  = min(top_k, index.ntotal)
    distances, indices = index.search(query_vec, actual_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx < 0:
            continue
        r = _make_result(rank, int(idx), metadata, float(dist), "dinov2")
        if r:
            results.append(r)

    return results


# ── Режим 2: DINOv2 + Depth ───────────────────────────────────────────────────

def search_hybrid(
    query_image: Image.Image,
    dino_index,
    depth_index,
    metadata: dict[int, str],
    top_k: int = TOP_K,
    dino_weight: float = 0.35,
    depth_weight: float = 0.65,
    candidates_k: int = 30,
) -> list[dict]:
    """Гибридный поиск: DINOv2 (35%) + Depth Anything V2 (65%)."""
    from app.depth_extractor import extract_depth_embedding

    dino_vec = extract_embedding(query_image).reshape(1, -1)
    actual_cand = min(candidates_k, dino_index.ntotal)
    dino_dists, dino_ids = dino_index.search(dino_vec, actual_cand)
    dino_dists = dino_dists[0]
    dino_ids   = dino_ids[0]

    depth_vec = extract_depth_embedding(query_image).reshape(1, -1)

    candidate_depth_vecs = np.stack([
        depth_index.reconstruct(int(idx))
        if (idx >= 0 and int(idx) < depth_index.ntotal)
        else np.zeros(depth_index.d, dtype=np.float32)
        for idx in dino_ids
    ], axis=0)

    depth_scores = (candidate_depth_vecs @ depth_vec.T).flatten()
    combined = dino_weight * np.clip(dino_dists, 0, 1) + depth_weight * np.clip(depth_scores, 0, 1)
    order = np.argsort(-combined)

    results = []
    rank = 1
    for i in order:
        idx = int(dino_ids[i])
        if idx < 0 or rank > top_k:
            continue
        dino_s   = float(np.clip(dino_dists[i], 0.0, 1.0))
        depth_s  = float(np.clip(depth_scores[i], 0.0, 1.0))
        hybrid_s = float(np.clip(combined[i], 0.0, 1.0))
        r = _make_result(rank, idx, metadata, hybrid_s, "hybrid", {
            "score_dino": round(dino_s * 100, 1),
            "score_depth": round(depth_s * 100, 1),
        })
        if r:
            results.append(r)
            rank += 1
    return results


# ── Режим 3: Depth only ─────────────────────────────────────────────────────────

def search_depth_only(
    query_image: Image.Image,
    depth_index,
    metadata: dict[int, str],
    top_k: int = TOP_K,
) -> list[dict]:
    """Поиск только по Depth Anything V2 (чистая геометрия без DINOv2)."""
    from app.depth_extractor import extract_depth_embedding

    depth_vec = extract_depth_embedding(query_image).reshape(1, -1)
    actual_k  = min(top_k, depth_index.ntotal)
    
    # В FAISS с IP (Inner Product) dist - это косинусное сходство.
    distances, indices = depth_index.search(depth_vec, actual_k)
    distances = distances[0]
    indices   = indices[0]
    
    results = []
    for rank, (dist, idx) in enumerate(zip(distances, indices), start=1):
        if idx < 0:
            continue
        score = float(np.clip(dist, 0.0, 1.0))
        r = _make_result(rank, int(idx), metadata, score, "depth-only")
        if r:
            results.append(r)
            
    return results


# ── Секция YOLO Rerank ────────────────────────────────────────────────────────

def apply_yolo_rerank(
    results: list[dict],
    query_yolo: dict,
    yolo_metadata: dict,
) -> list[dict]:
    """
    Применяет мягкий фильтр через множитель: бонус за совпадение секторов окон/дверей, штраф за несовпадение.
    """
    if not query_yolo or not yolo_metadata:
        return results

    qw = set(query_yolo.get("windows", []))
    qd = set(query_yolo.get("doors", []))

    for r in results:
        fname = r["filename"]
        cand_yolo = yolo_metadata.get(fname, {})
        cw = set(cand_yolo.get("windows", []))
        cd = set(cand_yolo.get("doors", []))

        spatial_multiplier = 1.0

        # Бонус за окна
        if qw:
            intersection_w = qw.intersection(cw)
            if intersection_w:
                spatial_multiplier += 0.10 * len(intersection_w)
            elif cw:
                spatial_multiplier -= 0.05 * len(qw)  # Окна есть, но не там
            else:
                spatial_multiplier -= 0.05 * len(qw)  # Строгий штраф: мы ищем окно, а YOLO его вообще не нашел

        # Бонус за двери
        if qd:
            intersection_d = qd.intersection(cd)
            if intersection_d:
                spatial_multiplier += 0.10 * len(intersection_d)
            elif cd:
                spatial_multiplier -= 0.05 * len(qd)
            else:
                spatial_multiplier -= 0.05 * len(qd)

        if spatial_multiplier != 1.0:
            old_score = r.get("score", 0.0)
            new_score = float(np.clip(old_score * spatial_multiplier, 0.0, 1.0))
            r["score"] = round(new_score, 4)
            r["score_pct"] = round(new_score * 100, 1)
            r["yolo_bonus"] = round(spatial_multiplier, 3)
            r["mode"] = r["mode"] + " + YOLO"

    # Пересортировка
    results.sort(key=lambda x: x["score"], reverse=True)

    # Обновление rank
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    return results
