"""Modelos ORM: telemetría e historial de predicciones.

No hay ventana ni purga: todo el historial queda en PostgreSQL.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

# BigInteger en PostgreSQL; Integer en SQLite (para que autoincrement funcione).
PK_TYPE = BigInteger().with_variant(Integer, "sqlite")
# JSONB en PostgreSQL; JSON genérico en SQLite.
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TelemetryRecord(Base):
    """Una muestra de telemetría enviada por un robot."""

    __tablename__ = "telemetry"

    id: Mapped[int] = mapped_column(PK_TYPE, primary_key=True, autoincrement=True)
    robot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    fleet_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)

    # Marca de tiempo de la muestra (según el cliente) y de recepción (servidor).
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    battery_pct: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[Optional[str]] = mapped_column(String(16))  # etiqueta real, si el simulador la envía

    # Muestra completa tal como llegó; permite añadir variables sin migrar.
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)

    __table_args__ = (Index("ix_telemetry_robot_ts", "robot_id", "ts"),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "robot_id": self.robot_id,
            "fleet_id": self.fleet_id,
            "ts": self.ts.isoformat() if self.ts else None,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "battery_pct": self.battery_pct,
            "status": self.status,
            **(self.payload or {}),
        }


class PredictionRecord(Base):
    """Resultado de una inferencia de POST /predict."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(PK_TYPE, primary_key=True, autoincrement=True)
    robot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    telemetry_id: Mapped[Optional[int]] = mapped_column(
        PK_TYPE, ForeignKey("telemetry.id", ondelete="SET NULL")
    )
    predicted_status: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    __table_args__ = (Index("ix_predictions_robot_created", "robot_id", "created_at"),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "predicted_status": self.predicted_status,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "inference_ms": self.inference_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }