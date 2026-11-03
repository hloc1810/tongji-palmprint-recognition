"""FastAPI Web Backend for Tongji Contactless Palmprint Recognition.

Serves the Tongji_Palmprint_UI.html frontend and provides real-time
inference endpoints using the project's frozen CompCode models (CRC_RLS, Linear SVM, 1-NN).
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from palmprint.evaluation.metrics import ranking
from palmprint.features.compcode import CompCodeConfig, CompCodeExtractor

WS = Path(__file__).resolve().parents[1]

# Global application cache
MODELS: dict[str, Any] = {}
EXTRACTOR: CompCodeExtractor | None = None
FEATURE_CONFIG: dict[str, Any] = {}
SAMPLES: list[dict[str, Any]] = []
SAMPLE_IMAGES: dict[int, bytes] = {}

MODEL_METADATA = {
    "crc": {
        "id": "crc",
        "name": "CRC_RLS (M3)",
        "full_name": "CRC_RLS (M3 · Rank-1 = 98.27% · Đề xuất)",
        "score_kind": "reconstruction_residual",
        "score_kind_label": "Phần dư tái tạo (Residual)",
        "rank1": "98.27%",
        "rank5": "99.25%",
        "latency_est": "15.5 ms",
        "description": "Biểu diễn cộng tác CRC ridge, chuẩn Euclidean unnormalized residual.",
    },
    "svm": {
        "id": "svm",
        "name": "Linear SVM (M2)",
        "full_name": "Linear SVM (M2 · Rank-1 = 97.37% · Nhanh nhất)",
        "score_kind": "negative_margin",
        "score_kind_label": "Khoảng cách lề âm (-Margin)",
        "rank1": "97.37%",
        "rank5": "98.80%",
        "latency_est": "3.71 ms",
        "description": "LinearSVC L2 squared-hinge OvR với synthetic intercept.",
    },
    "knn": {
        "id": "knn",
        "name": "1-NN Baseline (M1)",
        "full_name": "1-NN Baseline (M1 · Rank-1 = 94.43% · k=1)",
        "score_kind": "squared_euclidean",
        "score_kind_label": "Khoảng cách Euclidean bình phương",
        "rank1": "94.43%",
        "rank5": "96.95%",
        "latency_est": "11.6 ms",
        "description": "K-láng giềng k=1 nearest template theo gallery Session 1.",
    },
}

SAMPLE_DEFINITIONS = [
    (1, "00001.bmp", 1, "Mẫu 1 (Palm ID 0001)"),
    (2, "00415.bmp", 42, "Mẫu 2 (Palm ID 0042)"),
    (3, "01275.bmp", 128, "Mẫu 3 (Palm ID 0128)"),
    (4, "02995.bmp", 300, "Mẫu 4 (Palm ID 0300)"),
    (5, "04265.bmp", 427, "Mẫu 5 (Palm ID 0427)"),
    (6, "05985.bmp", 599, "Mẫu 6 (Palm ID 0599)"),
]


def load_all_artifacts() -> None:
    """Preload models, feature extractor, and sample probes into memory."""
    global EXTRACTOR, FEATURE_CONFIG, MODELS, SAMPLE_IMAGES, SAMPLES

    for name in ("crc", "svm", "knn"):
        path = WS / "artifacts" / f"{name}.joblib"
        if not path.exists():
            continue
        bundle = joblib.load(path)
        MODELS[name] = bundle["model"]
        if not FEATURE_CONFIG and "feature_config" in bundle:
            FEATURE_CONFIG = bundle["feature_config"]

    if FEATURE_CONFIG:
        EXTRACTOR = CompCodeExtractor(CompCodeConfig(**FEATURE_CONFIG))

    SAMPLES.clear()
    SAMPLE_IMAGES.clear()
    for sid, fname, true_pid, label in SAMPLE_DEFINITIONS:
        img_path = WS / "data/raw/session2" / fname
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, buf = cv2.imencode(".png", img)
        png_bytes = buf.tobytes()
        SAMPLE_IMAGES[sid] = png_bytes
        SAMPLES.append(
            {
                "id": sid,
                "name": label,
                "filename": f"session2_{fname}",
                "palm_id": true_pid,
                "true_palm_id": true_pid,
                "session": "session2",
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "image_url": f"/api/sample-image/{sid}",
            }
        )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODELS or not SAMPLES:
        load_all_artifacts()
    yield


# Eager preload on import so all endpoints have models and samples immediately
load_all_artifacts()

app = FastAPI(
    title="Tongji Contactless Palmprint Recognition API",
    description="Backend API for palmprint identification on Tongji Dataset (600 Palm IDs).",
    version="1.0.0",
    lifespan=lifespan,
)


# Custom CORS middleware to guarantee all origins including null (file:///) work reliably
@app.middleware("http")
async def add_cors_headers(request, call_next):
    if request.method == "OPTIONS":
        response = Response(status_code=204)
    else:
        response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def root():
    ui_path = WS / "Tongji_Palmprint_UI.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="Tongji_Palmprint_UI.html not found")
    return FileResponse(ui_path, media_type="text/html")


@app.get("/Tongji_Palmprint_UI.html")
async def serve_ui():
    ui_path = WS / "Tongji_Palmprint_UI.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="Tongji_Palmprint_UI.html not found")
    return FileResponse(ui_path, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "models": {k: {"loaded": k in MODELS, **MODEL_METADATA[k]} for k in MODEL_METADATA},
        "database": "Tongji Contactless Palmprint (600 Palm IDs, 12,000 images)",
        "samples_count": len(SAMPLES),
    }


@app.get("/api/models")
async def get_models():
    return list(MODEL_METADATA.values())


@app.get("/api/samples")
async def get_samples():
    return SAMPLES


@app.get("/api/sample-image/{sample_id}")
async def get_sample_image(sample_id: int):
    if not SAMPLE_IMAGES:
        load_all_artifacts()
    if sample_id not in SAMPLE_IMAGES:
        raise HTTPException(status_code=404, detail="Sample image not found")
    return Response(
        content=SAMPLE_IMAGES[sample_id],
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    model: str = Form("crc"),
):
    if not MODELS or EXTRACTOR is None:
        load_all_artifacts()

    chosen_model = model.lower().strip()
    if chosen_model not in MODELS:
        chosen_model = "crc"
    if chosen_model not in MODELS:
        raise HTTPException(status_code=500, detail="Models not loaded")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file payload")

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Format unsupported.")

    req_size = EXTRACTOR.config.image_size
    if img_gray.shape[0] != req_size or img_gray.shape[1] != req_size:
        img_gray = cv2.resize(img_gray, (req_size, req_size), interpolation=cv2.INTER_AREA)

    t0 = time.perf_counter()
    feature = EXTRACTOR.transform(img_gray)
    mdl = MODELS[chosen_model]
    scores = mdl.scores(feature[None, :])
    ids, vals = ranking(scores, mdl.classes_)
    latency_ms = (time.perf_counter() - t0) * 1000

    predicted_id = int(ids[0, 0])
    best_score = float(vals[0, 0])

    # Infer ground truth label from filename if present
    filename = (file.filename or "").lower()
    true_palm_id: int | None = None
    for sample in SAMPLES:
        if sample["filename"] in filename or f"{sample['palm_id']:04d}" in filename:
            true_palm_id = sample["palm_id"]
            break

    if true_palm_id is None:
        # Check standard Tongji naming patterns like 00001.bmp or palm_42
        for part in filename.replace(".", "_").split("_"):
            if part.isdigit() and len(part) == 5:
                idx = int(part)
                if 1 <= idx <= 6000:
                    true_palm_id = (idx - 1) // 10 + 1
                    break
            elif part.startswith("palm") and part[4:].isdigit():
                true_palm_id = int(part[4:])
                break

    is_correct = (predicted_id == true_palm_id) if true_palm_id is not None else None

    meta = MODEL_METADATA[chosen_model]
    top5_list = [
        {
            "rank": i + 1,
            "palm_id": int(ids[0, i]),
            "score": round(float(vals[0, i]), 4),
        }
        for i in range(min(5, ids.shape[1]))
    ]

    return JSONResponse(
        content={
            "palm_id": predicted_id,
            "predicted_palm_id": predicted_id,
            "true_palm_id": true_palm_id,
            "is_correct": is_correct,
            "score": round(best_score, 4),
            "score_kind": meta["score_kind"],
            "score_kind_label": meta["score_kind_label"],
            "latency_ms": round(latency_ms, 2),
            "model": chosen_model,
            "model_name": meta["full_name"],
            "top5": top5_list,
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
