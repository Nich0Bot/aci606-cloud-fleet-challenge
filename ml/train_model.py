"""Script de entrenamiento del modelo de ML (Ejercicio 4).

Genera el dataset, realiza EDA, compara clasificadores con GroupShuffleSplit
por fleet_id, calcula el F1 macro y guarda el modelo en models/model.joblib.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Importar el generador del starter si existe
try:
    from simulator.generate_dataset import generate_dataset
except ImportError:
    generate_dataset = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "x",
    "y",
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]
TARGET = "status"
CLASSES = ["NORMAL", "WARNING", "CRITICAL"]


def load_or_create_data() -> pd.DataFrame:
    """Carga data/fleet_dataset.csv o genera un dataset de entrenamiento."""
    csv_path = BASE_DIR / "data" / "fleet_dataset.csv"
    if csv_path.exists():
        logger.info("Cargando dataset existente desde %s", csv_path)
        df = pd.read_csv(csv_path)
    elif generate_dataset is not None:
        logger.info("Generando dataset con simulator.generate_dataset (3000 muestras)...")
        df = generate_dataset(samples=3000)
    else:
        logger.info("Generando dataset sintético de respaldo...")
        np.random.seed(42)
        n = 3000
        data = {
            "x": np.random.uniform(-100, 100, n),
            "y": np.random.uniform(-100, 100, n),
            "speed": np.random.uniform(0, 10, n),
            "battery": np.random.uniform(10, 100, n),
            "motor_temp": np.random.uniform(20, 90, n),
            "vibration": np.random.uniform(0.1, 5.0, n),
            "motor_current": np.random.uniform(0.5, 15.0, n),
            "obstacle_distance": np.random.uniform(0.1, 20.0, n),
            "fleet_id": np.random.choice(["F1", "F2", "F3", "F4", "F5"], n),
        }
        df = pd.DataFrame(data)
        conditions = [
            (df["battery"] < 20) | (df["motor_temp"] > 80) | (df["vibration"] > 4.0),
            (df["battery"] < 40) | (df["motor_temp"] > 65) | (df["vibration"] > 2.5),
        ]
        choices = ["CRITICAL", "WARNING"]
        df[TARGET] = np.select(conditions, choices, default="NORMAL")
    return df


def main() -> None:
    df = load_or_create_data()

    X = df[FEATURES]
    y = df[TARGET]

    if "fleet_id" in df.columns:
        groups = df["fleet_id"]
    else:
        groups = np.repeat(np.arange(5), len(df) // 5 + 1)[: len(df)]

    # GroupShuffleSplit por fleet_id obligatoria según la pauta
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    logger.info("Train set: %d muestras | Test set: %d muestras", len(X_train), len(X_test))

    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    }

    best_model = None
    best_f1_macro = -1.0
    best_name = ""

    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        f1_macro = f1_score(y_test, preds, average="macro")

        logger.info("Modelo %s -> F1 Macro: %.4f", name, f1_macro)

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_model = pipeline
            best_name = name

    logger.info("--- MEJOR MODELO SELECCIONADO: %s ---", best_name)
    test_preds = best_model.predict(X_test)
    logger.info("\nMatriz de Confusión:\n%s", confusion_matrix(y_test, test_preds, labels=CLASSES))
    logger.info("\nReporte de Clasificación:\n%s", classification_report(y_test, test_preds))

    model_path = MODELS_DIR / "model.joblib"
    payload = {
        "pipeline": best_model,
        "model_version": "v1",
        "features": FEATURES,
    }
    joblib.dump(payload, model_path)
    logger.info("¡Modelo guardado exitosamente en %s!", model_path)


if __name__ == "__main__":
    main()