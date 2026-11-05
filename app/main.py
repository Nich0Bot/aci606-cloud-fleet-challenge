"""API del Cloud Fleet Challenge: FastAPI + PostgreSQL (Neon) + modelo de ML."""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db import SessionLocal, check_db, get_db, init_db
from app.models import PredictionRecord, TelemetryRecord
from app.schemas import (
    MODEL_FEATURES_DEFAULT,
    PredictionOut,
    PredictRequest,
    PredictResponse,
    TelemetryAck,
    TelemetryIn,
    TelemetryOut,
)

logger = logging.getLogger("fleet")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_PATH = BASE_DIR / "dashboard" / "index.html"
VALID_STATUSES = {"NORMAL", "WARNING", "CRITICAL"}


# ---------------------------------------------------------------------------
# Métricas operacionales (en memoria; se reinician con el servicio)
# ---------------------------------------------------------------------------
def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)
    return {
        "count": n,
        "avg": round(sum(ordered) / n, 3) if n else 0.0,
        "p50": round(_percentile(ordered, 50), 3),
        "p95": round(_percentile(ordered, 95), 3),
        "max": round(ordered[-1], 3) if n else 0.0,
    }


class Metrics:
    def __init__(self, max_samples: int = 5000) -> None:
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.started_at = datetime.now(timezone.utc)
        self._latencies: deque[float] = deque(maxlen=max_samples)
        self._predict_latencies: deque[float] = deque(maxlen=max_samples)
        self._inference_ms: deque[float] = deque(maxlen=max_samples)
        self.requests_total = 0
        self.requests_telemetry = 0
        self.requests_predict = 0
        self.errors_4xx = 0
        self.errors_5xx = 0

    def record(self, method: str, path: str, status: int, ms: float) -> None:
        with self._lock:
            self.requests_total += 1
            self._latencies.append(ms)
            if method == "POST" and path == "/telemetry":
                self.requests_telemetry += 1
            elif method == "POST" and path == "/predict":
                self.requests_predict += 1
                self._predict_latencies.append(ms)
            if 400 <= status < 500:
                self.errors_4xx += 1
            elif status >= 500:
                self.errors_5xx += 1

    def record_inference(self, ms: float) -> None:
        with self._lock:
            self._inference_ms.append(ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_seconds": round(time.monotonic() - self._t0, 1),
                "started_at": self.started_at.isoformat(),
                "requests_total": self.requests_total,
                "requests_telemetry": self.requests_telemetry,
                "requests_predict": self.requests_predict,
                "errors_4xx": self.errors_4xx,
                "errors_5xx": self.errors_5xx,
                "errors_total": self.errors_4xx + self.errors_5xx,
                "latency_ms": _stats(list(self._latencies)),
                "predict_latency_ms": _stats(list(self._predict_latencies)),
                "inference_ms": _stats(list(self._inference_ms)),
            }


metrics = Metrics()


# ---------------------------------------------------------------------------
# Modelo de ML (se carga una vez al iniciar)
# ---------------------------------------------------------------------------
@dataclass
class ModelState:
    pipeline: Any = None
    version: str = "v1"
    features: list[str] = field(default_factory=lambda: list(MODEL_FEATURES_DEFAULT))
    path: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None


model_state = ModelState()


def load_model() -> None:
    raw = os.getenv("MODEL_PATH", "models/model.joblib")
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        logger.error("Modelo no encontrado en %s (¿está versionado con git add -f?)", path)
        return

    try:
        obj = joblib.load(path)
    except Exception:  # noqa: BLE001
        logger.exception("No se pudo cargar el modelo desde %s", path)
        return

    version = os.getenv("MODEL_VERSION", "v1")
    features: Optional[list[str]] = None

    # Formato opcional: {"pipeline": ..., "model_version": "v2", "features": [...]}
    if isinstance(obj, dict):
        version = str(obj.get("model_version", version))
        features = obj.get("features")
        obj = obj["pipeline"] if "pipeline" in obj else obj.get("model")

    if obj is None or not hasattr(obj, "predict"):
        logger.error("El archivo %s no contiene un modelo con método predict()", path)
        return

    if features is None:
        names = getattr(obj, "feature_names_in_", None)
        features = list(names) if names is not None else list(MODEL_FEATURES_DEFAULT)

    unknown = set(features) - set(TelemetryIn.model_fields)
    if unknown:
        logger.error("El modelo espera columnas que el contrato no entrega: %s", sorted(unknown))
        return

    model_state.pipeline = obj
    model_state.version = version
    model_state.features = list(features)
    model_state.path = str(path)
    logger.info("Modelo cargado: %s (version=%s, features=%s)", path, version, features)


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()      # si la BD no responde, el deploy falla rápido y visible
    load_model()
    yield


app = FastAPI(title="ACI606 Cloud Fleet Challenge", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        metrics.record(request.method, request.url.path, status, (time.perf_counter() - start) * 1000)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    """Siempre 422. No se devuelve 'input': un NaN allí rompería la serialización (-> 500)."""
    detail = [
        {"loc": list(e.get("loc", [])), "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    db_ok = check_db()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": db_ok,
        "model_loaded": model_state.loaded,
        "model_version": model_state.version if model_state.loaded else None,
        "uptime_seconds": round(time.monotonic() - metrics._t0, 1),
    }


@app.post("/telemetry", response_model=TelemetryAck)
def ingest_telemetry(sample: TelemetryIn, db: Session = Depends(get_db)):
    data = sample.model_dump()
    
    # Extraemos los campos principales que son columnas directas en la tabla
    robot_id = data.pop("robot_id")
    fleet_id = data.pop("fleet_id", None)
    data.pop("timestamp", None)  # Removemos el timestamp de la entrada
    
    # El resto del diccionario ('x', 'y', 'speed', etc.) se guarda dentro de 'payload'
    record = TelemetryRecord(
        robot_id=robot_id,
        fleet_id=fleet_id,
        payload=data
    )
    
    try:
        db.add(record)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Error al guardar telemetría")
        raise HTTPException(status_code=503, detail="Base de datos no disponible")
        
    return TelemetryAck(id=record.id, robot_id=record.robot_id)


@app.get("/robots", response_model=list[TelemetryOut])
def list_robots(db: Session = Depends(get_db)):
    """Última muestra de cada robot (una sola consulta con función ventana)."""
    rn = (
        func.row_number()
        .over(
            partition_by=TelemetryRecord.robot_id,
            order_by=(TelemetryRecord.ts.desc(), TelemetryRecord.id.desc()),
        )
        .label("rn")
    )

    ranked = select(TelemetryRecord.id.label("id"), rn).subquery()
    stmt = (
        select(TelemetryRecord)
        .join(ranked, TelemetryRecord.id == ranked.c.id)
        .where(ranked.c.rn == 1)
        .order_by(TelemetryRecord.robot_id)
    )
    try:
        records = db.scalars(stmt).all()
        result = []
        for r in records:
            payload_data = r.payload if isinstance(r.payload, dict) else {}
            
            # Construimos el objeto con valores por defecto para evitar ValidationErrors
            item = {
                "robot_id": r.robot_id,
                "fleet_id": r.fleet_id or "F1",
                "timestamp": r.ts.timestamp() if r.ts else 0.0,
                "x": float(payload_data.get("x", 0.0)),
                "y": float(payload_data.get("y", 0.0)),
                "speed": float(payload_data.get("speed", 0.0)),
                "battery": float(payload_data.get("battery", 100.0)),
                "motor_temp": float(payload_data.get("motor_temp", 0.0)),
                "vibration": float(payload_data.get("vibration", 0.0)),
                "motor_current": float(payload_data.get("motor_current", 0.0)),
                "obstacle_distance": float(payload_data.get("obstacle_distance", 0.0)),
            }
            result.append(item)
        return result
    except Exception:
        logger.exception("Error al consultar /robots")
        raise HTTPException(status_code=503, detail="Base de datos no disponible")


@app.get("/robots/{robot_id}/history", response_model=list[TelemetryOut])
def robot_history(robot_id: str, db: Session = Depends(get_db)):
    """Historial COMPLETO del robot (sin límite), en orden cronológico."""
    stmt = (
        select(TelemetryRecord)
        .where(TelemetryRecord.robot_id == robot_id)
        .order_by(TelemetryRecord.timestamp.asc(), TelemetryRecord.id.asc())
    )
    try:
        rows = db.scalars(stmt).all()
    except SQLAlchemyError:
        logger.exception("Error al consultar historial")
        raise HTTPException(status_code=503, detail="Base de datos no disponible")
    if not rows:
        raise HTTPException(status_code=404, detail=f"Robot '{robot_id}' sin registros")
    return rows


def _save_prediction(result: PredictResponse) -> None:
    """Guarda la predicción en segundo plano (no suma latencia a /predict)."""
    try:
        with SessionLocal() as session:
            session.add(PredictionRecord(**result.model_dump()))
            session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("No se pudo guardar la predicción")


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, background: BackgroundTasks):
    if not model_state.loaded:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    start = time.perf_counter()
    try:
        data = payload.model_dump()
        row = pd.DataFrame([[data[c] for c in model_state.features]], columns=model_state.features)
        label = str(model_state.pipeline.predict(row)[0]).upper()
        if hasattr(model_state.pipeline, "predict_proba"):
            confidence = float(model_state.pipeline.predict_proba(row).max())
        else:
            confidence = 1.0
        if label not in VALID_STATUSES:
            raise ValueError(f"El modelo devolvió una clase inesperada: {label!r}")
    except Exception:  # noqa: BLE001
        logger.exception("Error durante la inferencia")
        raise HTTPException(status_code=500, detail="Error de inferencia")
    inference_ms = (time.perf_counter() - start) * 1000

    result = PredictResponse(
        robot_id=payload.robot_id,
        predicted_status=label,
        confidence=round(confidence, 4),
        model_version=model_state.version,
        inference_ms=round(inference_ms, 3),
    )
    metrics.record_inference(inference_ms)
    background.add_task(_save_prediction, result)
    return result


@app.get("/predictions/latest", response_model=list[PredictionOut])
def latest_predictions(db: Session = Depends(get_db)):
    """Extra (no está en el contrato): última predicción por robot, para el panel."""
    rn = (
        func.row_number()
        .over(
            partition_by=PredictionRecord.robot_id,
            order_by=(PredictionRecord.created_at.desc(), PredictionRecord.id.desc()),
        )
        .label("rn")
    )
    ranked = select(PredictionRecord.id.label("id"), rn).subquery()
    stmt = (
        select(PredictionRecord)
        .join(ranked, PredictionRecord.id == ranked.c.id)
        .where(ranked.c.rn == 1)
        .order_by(PredictionRecord.robot_id)
    )
    try:
        return db.scalars(stmt).all()
    except SQLAlchemyError:
        logger.exception("Error al consultar predicciones")
        raise HTTPException(status_code=503, detail="Base de datos no disponible")


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    data = metrics.snapshot()
    data["model_loaded"] = model_state.loaded
    data["model_version"] = model_state.version if model_state.loaded else None
    return data


_FALLBACK_DASHBOARD = """<!doctype html><meta charset="utf-8"><title>Fleet</title>
<body style="font-family:sans-serif"><h1>Panel no encontrado</h1>
<p>Falta <code>dashboard/index.html</code> en el repositorio.</p></body>"""


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    if DASHBOARD_PATH.exists():
        return FileResponse(DASHBOARD_PATH, media_type="text/html")
    return HTMLResponse(_FALLBACK_DASHBOARD, status_code=200)