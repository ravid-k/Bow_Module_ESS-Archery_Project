from __future__ import annotations

import base64
import importlib.util
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

APP_DIR = Path(__file__).resolve().parent
VISION_MODULE_PATH = Path(
    os.getenv(
        "ESS_VISION_MODULE",
        str(APP_DIR / "vision_v0_2_arrow_shaft.py"),
    )
)

if not VISION_MODULE_PATH.exists():
    fallback = APP_DIR / "vision_v0_5_shot_coordinates_fixed.py"
    if fallback.exists():
        VISION_MODULE_PATH = fallback


def _load_vision_module(path: Path):
    if not path.exists():
        raise RuntimeError(
            f"Vision module not found: {path}. "
            "Put V0.5 in this folder or set ESS_VISION_MODULE."
        )
    spec = importlib.util.spec_from_file_location("ess_archery_vision", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load vision module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vision = _load_vision_module(VISION_MODULE_PATH)

app = FastAPI(
    title="ESS-ARCHERY Vision API",
    version="0.5",
    description="ArUco + arrow impact + target coordinate analysis for ESS-ARCHERY.",
)

# Prototype/demo configuration. Tighten to your final Cloudflare origin later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "ESS-ARCHERY Vision API",
        "vision_version": "V0.5",
        "module": VISION_MODULE_PATH.name,
    }


def _save_upload(upload: UploadFile, destination: Path) -> None:
    with destination.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _analyze(reference_path: Path, current_path: Path, output_dir: Path, target_diameter_cm: float):
    reference = vision.prepare_image(reference_path)

    if len(reference["markers"]) < 4:
        return {
            "success": False,
            "status": "REFERENCE_RETAKE",
            "message": "Reference image must show all four ArUco markers (IDs 0,1,2,3).",
            "aruco_markers": len(reference["markers"]),
            "marker_ids": reference.get("marker_ids", []),
        }

    yellow = vision.detect_yellow_ellipse(reference["proc"])
    if yellow is None:
        return {
            "success": False,
            "status": "REFERENCE_TARGET_NOT_FOUND",
            "message": "Yellow target reference could not be detected. Retake the reference image.",
        }

    yellow_ellipse = yellow["ellipse"]
    target_mask = vision.make_target_mask(
        reference["proc"].shape,
        yellow_ellipse,
        scale_factor=5.12,
    )

    result = vision.process_one(
        reference,
        yellow_ellipse,
        target_mask,
        current_path,
        output_dir,
        target_diameter_cm=target_diameter_cm,
    )

    if result["status"] != "ARROW" or not result.get("coordinates"):
        return {
            "success": False,
            "status": result["status"],
            "message": (
                "Arrow/impact was not detected confidently. Retake the target photo or use Manual mode."
                if result["status"] != "RETAKE"
                else "Fewer than three ArUco markers were usable. Retake the target photo."
            ),
            "aruco_markers": result.get("markers"),
            "shaft_score": result.get("score"),
        }

    coords = result["coordinates"]
    result_image_path = output_dir / f"{current_path.stem}_04_result.jpg"
    result_image_data_url = None
    if result_image_path.exists():
        encoded = base64.b64encode(result_image_path.read_bytes()).decode("ascii")
        result_image_data_url = f"data:image/jpeg;base64,{encoded}"

    return {
        "success": True,
        "status": "ARROW",
        "schema": "ESS_ARCHERY_SHOT_RESULT_V0.5_API",
        "auto_result": {
            "x_cm": round(float(coords["x_cm"]), 2),
            "y_cm": round(float(coords["y_cm"]), 2),
            "radial_cm": round(float(coords["radial_cm"]), 2),
            "radial_norm": round(float(coords["radial_norm"]), 5),
            "auto_ring": coords["auto_ring"],
            "x_norm": round(float(coords["x_norm"]), 6),
            "y_norm": round(float(coords["y_norm"]), 6),
        },
        "target": {
            "diameter_cm": float(target_diameter_cm),
            "radius_cm": float(target_diameter_cm) / 2.0,
        },
        "vision": {
            "aruco_markers": int(result.get("markers") or 0),
            "shaft_score": round(float(result.get("score") or 0), 4),
            "impact_mode": result.get("impact_mode") or "NORMAL",
            "confidence": result.get("vision_confidence") or "LOW",
        },
        "impact_current_px": {
            "x": int(result["impact"][0]),
            "y": int(result["impact"][1]),
        } if result.get("impact") else None,
        "impact_reference_px": {
            "x": round(float(result["impact_reference"][0]), 3),
            "y": round(float(result["impact_reference"][1]), 3),
        } if result.get("impact_reference") else None,
        "result_image_data_url": result_image_data_url,
        "manual_override_supported": True,
    }


@app.post("/analyze-shot")
async def analyze_shot(
    reference: UploadFile = File(...),
    current: UploadFile = File(...),
    target_diameter_cm: float = Form(60.0),
):
    if target_diameter_cm <= 0 or target_diameter_cm > 200:
        raise HTTPException(status_code=400, detail="Invalid target diameter.")

    allowed_prefixes = ("image/",)
    for upload, label in ((reference, "reference"), (current, "current")):
        content_type = upload.content_type or ""
        if not content_type.startswith(allowed_prefixes):
            raise HTTPException(status_code=400, detail=f"{label} must be an image file.")

    with tempfile.TemporaryDirectory(prefix="ess_archery_vision_") as tmp:
        tmp_path = Path(tmp)
        reference_suffix = Path(reference.filename or "reference.jpg").suffix or ".jpg"
        current_suffix = Path(current.filename or "current.jpg").suffix or ".jpg"
        reference_path = tmp_path / f"reference{reference_suffix}"
        current_path = tmp_path / f"current{current_suffix}"
        output_dir = tmp_path / "output"

        await run_in_threadpool(_save_upload, reference, reference_path)
        await run_in_threadpool(_save_upload, current, current_path)

        try:
            result = await run_in_threadpool(
                _analyze,
                reference_path,
                current_path,
                output_dir,
                float(target_diameter_cm),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Vision processing failed: {type(exc).__name__}: {exc}",
            ) from exc

        return result
