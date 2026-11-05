"""Esquemas Pydantic del contrato de la API (Ejercicio 4)."""
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["NORMAL", "WARNING", "CRITICAL"]

# ---------------------------------------------------------------------------
# Rangos físicos permitidos (generosos a propósito).
# AJUSTAR con el EDA: min/max observados en el dataset, con margen.
# Fuera de rango, NaN o inf  ->  HTTP 422 (4xx), nunca 5xx.
# ---------------------------------------------------------------------------
Timestamp = Annotated[float, Field(ge=0, le=1e13, allow_inf_nan=False)]
Position = Annotated[float, Field(ge=-1e6, le=1e6, allow_inf_nan=False)]
Speed = Annotated[float, Field(ge=-100, le=100, allow_inf_nan=False)]
Battery = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
MotorTemp = Annotated[float, Field(ge=-50, le=300, allow_inf_nan=False)]
Vibration = Annotated[float, Field(ge=0, le=1000, allow_inf_nan=False)]
MotorCurrent = Annotated[float, Field(ge=0, le=1000, allow_inf_nan=False)]
ObstacleDistance = Annotated[float, Field(ge=0, le=10000, allow_inf_nan=False)]

RobotId = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$"),
]

# Columnas que usa el modelo si el pipeline no trae feature_names_in_.
MODEL_FEATURES_DEFAULT = [
    "x",
    "y",
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]


class TelemetryIn(BaseModel):
    """Muestra de telemetría (POST /telemetry). Todos los campos son obligatorios."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    robot_id: RobotId
    timestamp: Timestamp
    x: Position
    y: Position
    speed: Speed
    battery: Battery
    motor_temp: MotorTemp
    vibration: Vibration
    motor_current: MotorCurrent
    obstacle_distance: ObstacleDistance


class PredictRequest(TelemetryIn):
    """Cuerpo de POST /predict: mismos campos que la telemetría."""


class TelemetryAck(BaseModel):
    status: Literal["ok"] = "ok"
    id: int
    robot_id: str


class TelemetryOut(BaseModel):
    """Muestra almacenada (GET /robots y GET /robots/{id}/history)."""

    model_config = ConfigDict(from_attributes=True)

    robot_id: str
    timestamp: float
    x: float
    y: float
    speed: float
    battery: float
    motor_temp: float
    vibration: float
    motor_current: float
    obstacle_distance: float
    received_at: datetime


class PredictResponse(BaseModel):
    """Respuesta obligatoria de POST /predict."""

    robot_id: str
    predicted_status: Status
    confidence: float
    model_version: str
    inference_ms: float


class PredictionOut(PredictResponse):
    """Predicción almacenada (GET /predictions/latest, para el panel)."""

    model_config = ConfigDict(from_attributes=True)

    created_at: datetime