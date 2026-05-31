"""
Esquema de features, defaults y transformación de entradas para inferencia.
Inferido de accidentes_cr_con_llm.csv y proyecto__3_grupo_b.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Raíz del proyecto (Proyecto3/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "clase_bin"

# 16 variables predictoras del modelo final (Con LLM)
FEATURE_COLUMNS = [
    "Kilometro_num",
    "Provincia",
    "Cantón",
    "region",
    "Rural_urbano",
    "Calzada_vertical",
    "Calzada_horizontal",
    "hora_grupo",
    "fin_semana",
    "estacion",
    "calzada_tipo",
    "estado_via",
    "clima_grupo",
    "tipo_colision",
    "severidad_tipo_LLM",
    "cobertura_emergencias_canton",
]

NUMERIC_FEATURES = [
    "Kilometro_num",
    "fin_semana",
    "severidad_tipo_LLM",
    "cobertura_emergencias_canton",
]

CATEGORICAL_FEATURES = [
    "Provincia",
    "Cantón",
    "region",
    "Rural_urbano",
    "Calzada_vertical",
    "Calzada_horizontal",
    "hora_grupo",
    "estacion",
    "calzada_tipo",
    "estado_via",
    "clima_grupo",
    "tipo_colision",
]

BINARY_FEATURES = ["fin_semana"]

# Tipo de control UI por variable
FEATURE_SCHEMA: dict[str, dict[str, Any]] = {
    "Kilometro_num": {"ui": "slider", "label": "Kilómetro en ruta"},
    "Provincia": {"ui": "dropdown", "label": "Provincia"},
    "Cantón": {"ui": "dropdown", "label": "Cantón"},
    "region": {"ui": "dropdown", "label": "Región"},
    "Rural_urbano": {"ui": "dropdown", "label": "Rural / Urbano"},
    "Calzada_vertical": {"ui": "dropdown", "label": "Calzada vertical"},
    "Calzada_horizontal": {"ui": "dropdown", "label": "Calzada horizontal"},
    "hora_grupo": {"ui": "dropdown", "label": "Franja horaria"},
    "fin_semana": {"ui": "selectbox", "label": "Fin de semana"},
    "estacion": {"ui": "dropdown", "label": "Estación"},
    "calzada_tipo": {"ui": "dropdown", "label": "Tipo de calzada"},
    "estado_via": {"ui": "dropdown", "label": "Estado de vía"},
    "clima_grupo": {"ui": "dropdown", "label": "Clima"},
    "tipo_colision": {"ui": "dropdown", "label": "Tipo de colisión"},
    "severidad_tipo_LLM": {"ui": "slider", "label": "Severidad tipo colisión (LLM)"},
    "cobertura_emergencias_canton": {"ui": "slider", "label": "Cobertura emergencias cantón (LLM)"},
}

CLASS_LABELS = {0: "Leve (sin muertos/graves)", 1: "Grave (muertos o heridos graves)"}


def get_data_path(filename: str = "accidentes_cr_con_llm.csv") -> Path:
    """Ruta al CSV principal dentro de data/."""
    return DATA_DIR / filename


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Carga el dataset enriquecido con LLM."""
    csv_path = path or get_data_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def load_reference_dataset(path: Path | None = None) -> pd.DataFrame:
    """Carga el subconjunto de referencia para drift."""
    ref_path = path or (DATA_DIR / "referencia.csv")
    if not ref_path.exists():
        raise FileNotFoundError(
            f"No se encontró referencia.csv en {ref_path}. "
            "Ejecute: python scripts/train_model.py"
        )
    return pd.read_csv(ref_path)


def compute_feature_defaults(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calcula valores por defecto para la UI a partir del dataset.
    Numéricas: mediana. Categóricas: moda. Binarias: moda.
    """
    defaults: dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        if col in NUMERIC_FEATURES:
            defaults[col] = float(df[col].median())
        else:
            mode = df[col].mode()
            defaults[col] = mode.iloc[0] if len(mode) else df[col].iloc[0]
    return defaults


def compute_feature_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Rangos min/max para sliders numéricos."""
    ranges: dict[str, dict[str, float]] = {}
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "step": 1.0 if col in ("fin_semana", "severidad_tipo_LLM", "cobertura_emergencias_canton") else 0.5,
        }
    return ranges


def get_categorical_options(df: pd.DataFrame) -> dict[str, list]:
    """Opciones únicas ordenadas por frecuencia para cada categórica."""
    options: dict[str, list] = {}
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        options[col] = counts.index.tolist()
    return options


def validate_inputs(
    user_inputs: dict[str, Any],
    categorical_options: dict[str, list],
) -> tuple[dict[str, Any], list[str]]:
    """
    Valida y corrige entradas del usuario.
    Retorna inputs corregidos y lista de advertencias.
    """
    cleaned = dict(user_inputs)
    warnings: list[str] = []

    for col in NUMERIC_FEATURES:
        if col not in cleaned:
            continue
        val = cleaned[col]
        if col in ("severidad_tipo_LLM", "cobertura_emergencias_canton"):
            clipped = int(np.clip(round(float(val)), 1, 10))
            if clipped != val:
                warnings.append(f"{col} ajustado al rango [1, 10].")
            cleaned[col] = clipped
        elif col == "fin_semana":
            cleaned[col] = int(np.clip(int(val), 0, 1))
        else:
            cleaned[col] = float(val)

    for col in CATEGORICAL_FEATURES:
        if col not in cleaned:
            continue
        val = str(cleaned[col])
        allowed = categorical_options.get(col, [])
        if allowed and val not in allowed:
            fallback = allowed[0]
            warnings.append(
                f"'{val}' no visto en entrenamiento para {col}. "
                f"Se usa '{fallback}'."
            )
            cleaned[col] = fallback

    return cleaned, warnings


def build_input_dataframe(user_inputs: dict[str, Any]) -> pd.DataFrame:
    """Construye un DataFrame de una fila con el orden correcto de columnas."""
    row = {col: user_inputs[col] for col in FEATURE_COLUMNS}
    return pd.DataFrame([row])


def transform_input(df: pd.DataFrame, preprocessor) -> np.ndarray:
    """Aplica el preprocessor entrenado (sin target)."""
    return preprocessor.transform(df[FEATURE_COLUMNS])


def get_transformed_feature_names(preprocessor) -> list[str]:
    """
    Nombres aproximados de features tras el ColumnTransformer.
    Numéricas conservan nombre; categóricas TargetEncoder produce una columna c/u.
    """
    names: list[str] = []
    for col in NUMERIC_FEATURES:
        names.append(col)
    for col in CATEGORICAL_FEATURES:
        names.append(col)
    return names
