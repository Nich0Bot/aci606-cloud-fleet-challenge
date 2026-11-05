"""Configuración de la base de datos (SQLAlchemy 2.x + psycopg 3).

En producción (Render) se usa PostgreSQL en Neon mediante DATABASE_URL.
En desarrollo local, si DATABASE_URL no está definida, se usa SQLite.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

LOCAL_FALLBACK_URL = "sqlite:///./local_dev.db"


class Base(DeclarativeBase):
    """Clase base de todos los modelos ORM."""


def normalize_database_url(raw: str) -> URL:
    """Adapta la URL entregada por Neon al driver psycopg 3.

    postgres://...      -> postgresql+psycopg://...
    postgresql://...    -> postgresql+psycopg://...
    Agrega sslmode=require si no viene especificado.
    """
    raw = raw.strip().strip("'\"")
    url = make_url(raw)

    if url.get_backend_name() in ("postgres", "postgresql"):
        url = url.set(drivername="postgresql+psycopg")
        if "sslmode" not in url.query:
            url = url.update_query_dict({"sslmode": "require"})
    return url


def _build_engine() -> Engine:
    raw_url = os.getenv("DATABASE_URL")

    if not raw_url:
        # Render define RENDER=true: allí jamás debemos caer a SQLite (disco efímero).
        if os.getenv("RENDER"):
            raise RuntimeError(
                "DATABASE_URL no está definida en Render. "
                "Configúrala en Environment con la cadena de conexión de Neon."
            )
        logger.warning("DATABASE_URL no definida: usando SQLite local (%s).", LOCAL_FALLBACK_URL)
        raw_url = LOCAL_FALLBACK_URL

    url = normalize_database_url(raw_url)

    if url.get_backend_name() == "sqlite":
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )

    return create_engine(
        url,
        pool_pre_ping=True,   # reconecta si Neon suspendió la base por inactividad
        pool_recycle=300,     # recicla conexiones antes de que Neon las cierre
        pool_size=5,
        max_overflow=5,
        pool_timeout=30,
        connect_args={
            "connect_timeout": 10,
            "prepare_threshold": None,  # compatible con el pooler (PgBouncer) de Neon
        },
    )


engine: Engine = _build_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Iterator[Session]:
    """Dependencia de FastAPI: una sesión por solicitud, siempre cerrada."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Crea las tablas si no existen. Llamar una vez al iniciar la app."""
    from app import models  # noqa: F401  (registra los modelos en Base.metadata)

    Base.metadata.create_all(bind=engine)


def check_db() -> bool:
    """Verifica conectividad con la base (útil para /health)."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Falla de conectividad con la base de datos")
        return False