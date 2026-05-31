"""
Carga de artefactos, predicción y explicabilidad (SHAP / importancias).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    MODELS_DIR,
    build_input_dataframe,
    get_transformed_feature_names,
    transform_input,
)


@dataclass
class ArtifactBundle:
    """Contenedor de artefactos cargados o estado degradado."""

    model: Any | None
    preprocessor: Any | None
    metadata: dict | None
    loaded: bool
    message: str


def load_metadata(path: Path | None = None) -> dict | None:
    """Carga model_metadata.json si existe."""
    meta_path = path or (MODELS_DIR / "model_metadata.json")
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def load_artifacts() -> ArtifactBundle:
    """
    Carga modelo, preprocessor y metadata.
    Retorna estado degradado con mensaje si faltan archivos.
    """
    model_path = MODELS_DIR / "modelo_final.pkl"
    prep_path = MODELS_DIR / "preprocessor.pkl"
    meta_path = MODELS_DIR / "model_metadata.json"

    missing = [
        p.name
        for p in (model_path, prep_path, meta_path)
        if not p.exists()
    ]
    if missing:
        return ArtifactBundle(
            model=None,
            preprocessor=None,
            metadata=None,
            loaded=False,
            message=(
                f"Artefactos faltantes: {', '.join(missing)}. "
                "Ejecute: python scripts/train_model.py"
            ),
        )

    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        metadata = load_metadata(meta_path)
        return ArtifactBundle(
            model=model,
            preprocessor=preprocessor,
            metadata=metadata,
            loaded=True,
            message="Modelo cargado correctamente.",
        )
    except Exception as exc:
        return ArtifactBundle(
            model=None,
            preprocessor=None,
            metadata=None,
            loaded=False,
            message=f"Error al cargar artefactos: {exc}",
        )


def predict(
    user_inputs: dict[str, Any],
    model,
    preprocessor,
) -> dict[str, Any]:
    """
    Realiza predicción sobre una fila de inputs validados.
    Retorna clase, probabilidades y etiqueta legible.
    """
    df = build_input_dataframe(user_inputs)
    X = transform_input(df, preprocessor)

    pred_class = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    prob_grave = float(proba[1])
    prob_leve = float(proba[0])

    return {
        "prediction": pred_class,
        "label": CLASS_LABELS[pred_class],
        "probability_grave": prob_grave,
        "probability_leve": prob_leve,
        "probabilities": {"leve": prob_leve, "grave": prob_grave},
    }


def explain_prediction_text(result: dict[str, Any]) -> str:
    """Genera explicación textual simple de la predicción."""
    pred = result["prediction"]
    p_grave = result["probability_grave"]
    if pred == 1:
        return (
            f"El modelo clasifica este accidente como **GRAVE** "
            f"(probabilidad de clase grave: {p_grave:.1%}). "
            "Se recomienda activar protocolos de respuesta prioritaria."
        )
    return (
        f"El modelo clasifica este accidente como **LEVE** "
        f"(probabilidad de clase leve: {result['probability_leve']:.1%}). "
        "No se detecta patrón de alta severidad según las variables ingresadas."
    )


def get_feature_importances(model, preprocessor) -> pd.DataFrame:
    """
    Retorna importancias del Random Forest alineadas con nombres de features.
    """
    names = get_transformed_feature_names(preprocessor)
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    imp = model.feature_importances_
    n = min(len(names), len(imp))
    df = pd.DataFrame({"feature": names[:n], "importance": imp[:n]})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _extract_shap_class1(shap_values: Any, n_features: int) -> np.ndarray:
    """
    Normaliza la salida de TreeExplainer a un vector 1D de floats (clase 1 = grave).

    Versiones recientes de SHAP devuelven ndarray con forma (muestras, features, clases);
    versiones antiguas devuelven lista [shap_clase0, shap_clase1] con forma (muestras, features).
  Si se toma shap_values[0] con forma (16, 2), cada celda es una lista de 2 valores
    y .abs() falla con: bad operand type for abs(): 'list'.
    """
    positive_class_index = 1

    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            return np.array([], dtype=float)
        # Lista por clase: usar índice 1 (grave) o la última si solo hay un canal
        class_idx = min(positive_class_index, len(shap_values) - 1)
        arr = np.asarray(shap_values[class_idx], dtype=float)
        if arr.ndim == 2:
            return np.ravel(arr[0, :n_features])
        return np.ravel(arr[:n_features])

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        class_idx = min(positive_class_index, arr.shape[2] - 1)
        return np.ravel(arr[0, :n_features, class_idx])
    if arr.ndim == 2:
        # (n_features, n_classes) — una muestra con SHAP por clase
        if arr.shape[1] >= 2 and arr.shape[0] >= n_features:
            class_idx = min(positive_class_index, arr.shape[1] - 1)
            return arr[:n_features, class_idx].astype(float)
        # (n_samples, n_features)
        if arr.shape[0] == 1:
            return np.ravel(arr[0, :n_features]).astype(float)
        if arr.shape[0] >= n_features and arr.shape[1] == 1:
            return np.ravel(arr[:n_features, 0]).astype(float)
    return np.ravel(arr)[:n_features].astype(float)


def _extract_shap_base_value(explainer: Any) -> float:
    """Valor base (expected value) para la clase positiva."""
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)):
        flat = np.asarray(ev, dtype=float).ravel()
        if flat.size >= 2:
            return float(flat[1])
        if flat.size == 1:
            return float(flat[0])
        return 0.0
    return float(ev)


def compute_shap_values(
    user_inputs: dict[str, Any],
    model,
    preprocessor,
    background: pd.DataFrame | None = None,
    max_background: int = 100,
) -> dict[str, Any] | None:
    """
    Calcula valores SHAP con TreeExplainer para una fila.
    Usa muestra de background para acelerar el cálculo.
    """
    try:
        import shap
    except ImportError:
        return None

    df = build_input_dataframe(user_inputs)
    X = transform_input(df, preprocessor)

    if background is not None and len(background) > 0:
        bg = background[FEATURE_COLUMNS].head(max_background)
        bg_t = transform_input(bg, preprocessor)
        explainer = shap.TreeExplainer(model, data=bg_t)
    else:
        explainer = shap.TreeExplainer(model)

    raw_shap = explainer.shap_values(X)
    names = get_transformed_feature_names(preprocessor)
    n = len(names)
    sv = _extract_shap_class1(raw_shap, n)
    n = min(n, len(sv))

    return {
        "features": names[:n],
        "shap_values": [float(v) for v in sv[:n]],
        "base_value": _extract_shap_base_value(explainer),
    }
