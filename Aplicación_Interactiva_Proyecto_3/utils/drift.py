"""
Detección de drift: alteraciones de dataset, PSI, KS, deltas y semáforo de riesgo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from utils.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)

# Pesos para drift score global (mayor peso en features críticas)
FEATURE_WEIGHTS: dict[str, float] = {
    "Kilometro_num": 1.2,
    "severidad_tipo_LLM": 1.5,
    "cobertura_emergencias_canton": 1.5,
    "fin_semana": 1.0,
}
DEFAULT_WEIGHT = 1.0

PSI_EPS = 1e-4
N_BINS = 10


def get_psi_status(psi: float) -> tuple[str, str, str]:
    """
    Retorna (color, emoji_label, mensaje) según umbrales PSI.
    VERDE: PSI < 0.10 | AMARILLO: 0.10-0.25 | ROJO: >= 0.25
    """
    if psi < 0.10:
        return "green", "VERDE", "Distribución estable"
    if psi < 0.25:
        return "yellow", "AMARILLO", "Existe evidencia moderada de drift"
    return "red", "ROJO", "Modelo fuera de dominio de entrenamiento"


def _safe_prop(series: pd.Series, value, epsilon: float = PSI_EPS) -> float:
    """Proporción suavizada para evitar log(0) en PSI."""
    n = len(series)
    if n == 0:
        return epsilon
    count = (series == value).sum()
    return max(count / n, epsilon)


def compute_psi_numeric(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = N_BINS,
) -> float:
    """
    Population Stability Index para variables numéricas.
    Bins definidos por cuantiles del dataset de referencia.
    """
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    if len(ref) < n_bins or len(cur) == 0:
        return 0.0

    try:
        bins = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
        if len(bins) < 2:
            bins = np.linspace(ref.min(), ref.max() + 1e-6, n_bins + 1)
    except Exception:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, PSI_EPS, None)
    cur_pct = np.clip(cur_pct, PSI_EPS, None)
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(max(psi, 0.0))


def compute_psi_categorical(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """PSI para variables categóricas basado en frecuencias relativas."""
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    categories = ref.value_counts(normalize=True).index.tolist()
    psi = 0.0
    for cat in categories:
        ref_pct = max((ref == cat).mean(), PSI_EPS)
        cur_pct = max((cur == cat).mean(), PSI_EPS)
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

    # Penalizar categorías nuevas en current
    new_cats = set(cur.unique()) - set(ref.unique())
    if new_cats:
        new_pct = sum((cur == c).mean() for c in new_cats)
        psi += new_pct * np.log(max(new_pct, PSI_EPS) / PSI_EPS)

    return float(max(psi, 0.0))


def compute_ks_numeric(reference: pd.Series, current: pd.Series) -> float:
    """Estadístico KS para numéricas."""
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    stat, _ = ks_2samp(ref, cur)
    return float(stat)


def compute_ks_categorical(reference: pd.Series, current: pd.Series) -> float:
    """Distancia máxima entre CDFs empíricas de categorías (aprox. KS)."""
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)
    all_cats = sorted(set(ref.unique()) | set(cur.unique()))
    if not all_cats:
        return 0.0
    ref_cdf = np.cumsum([(ref == c).mean() for c in all_cats])
    cur_cdf = np.cumsum([(cur == c).mean() for c in all_cats])
    return float(np.max(np.abs(cur_cdf - ref_cdf)))


def compute_mean_std_change(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[float, float]:
    """Cambio relativo de media y desviación estándar."""
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    ref_std = ref.std()
    if ref_std == 0 or np.isnan(ref_std):
        ref_std = 1.0
    delta_mean = (cur.mean() - ref.mean()) / ref_std
    delta_std = (cur.std() - ref.std()) / ref_std if len(cur) > 1 else 0.0
    return float(delta_mean), float(delta_std)


def compute_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """Calcula todas las métricas de drift para una columna."""
    ref_col = reference[column]
    cur_col = current[column]

    if column in NUMERIC_FEATURES:
        psi = compute_psi_numeric(ref_col, cur_col)
        ks = compute_ks_numeric(ref_col, cur_col)
        delta_mean, delta_std = compute_mean_std_change(ref_col, cur_col)
    else:
        psi = compute_psi_categorical(ref_col, cur_col)
        ks = compute_ks_categorical(ref_col, cur_col)
        delta_mean, delta_std = 0.0, 0.0

    color, status, message = get_psi_status(psi)
    return {
        "feature": column,
        "psi": psi,
        "ks": ks,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "status_color": color,
        "status_label": status,
        "status_message": message,
        "type": "numeric" if column in NUMERIC_FEATURES else "categorical",
    }


def compute_global_drift_score(drift_rows: list[dict[str, Any]]) -> float:
    """Promedio ponderado de PSI sobre todas las features."""
    if not drift_rows:
        return 0.0
    total_w = 0.0
    weighted = 0.0
    for row in drift_rows:
        w = FEATURE_WEIGHTS.get(row["feature"], DEFAULT_WEIGHT)
        weighted += row["psi"] * w
        total_w += w
    return float(weighted / total_w) if total_w else 0.0


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Reporte completo de drift para todas las features."""
    cols = features or [c for c in FEATURE_COLUMNS if c in reference.columns]
    rows = [compute_feature_drift(reference, current, c) for c in cols]
    global_score = compute_global_drift_score(rows)
    color, status, message = get_psi_status(global_score)
    return {
        "features": rows,
        "global_psi": global_score,
        "global_status_color": color,
        "global_status_label": status,
        "global_status_message": message,
    }


# ── Alteraciones del dataset (modo auditor) ─────────────────────────────


def apply_mean_shift(
    df: pd.DataFrame,
    columns: list[str],
    delta: float,
) -> pd.DataFrame:
    """Desplaza la media de columnas numéricas."""
    out = df.copy()
    for col in columns:
        if col in NUMERIC_FEATURES and col in out.columns:
            out[col] = out[col].astype(float) + delta
            if col in ("severidad_tipo_LLM", "cobertura_emergencias_canton"):
                out[col] = out[col].clip(1, 10).round().astype(int)
            elif col == "fin_semana":
                out[col] = out[col].clip(0, 1).round().astype(int)
    return out


def apply_std_inflation(
    df: pd.DataFrame,
    columns: list[str],
    factor: float,
) -> pd.DataFrame:
    """Aumenta la desviación estándar: x' = mean + (x - mean) * factor."""
    out = df.copy()
    for col in columns:
        if col in NUMERIC_FEATURES and col in out.columns:
            mean = out[col].astype(float).mean()
            out[col] = mean + (out[col].astype(float) - mean) * factor
            if col in ("severidad_tipo_LLM", "cobertura_emergencias_canton"):
                out[col] = out[col].clip(1, 10).round().astype(int)
            elif col == "fin_semana":
                out[col] = out[col].clip(0, 1).round().astype(int)
    return out


def apply_noise_injection(
    df: pd.DataFrame,
    columns: list[str],
    sigma: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """Inyecta ruido gaussiano en columnas numéricas."""
    rng = np.random.default_rng(random_state)
    out = df.copy()
    for col in columns:
        if col in NUMERIC_FEATURES and col in out.columns:
            noise = rng.normal(0, sigma, size=len(out))
            out[col] = out[col].astype(float) + noise
            if col in ("severidad_tipo_LLM", "cobertura_emergencias_canton"):
                out[col] = out[col].clip(1, 10).round().astype(int)
            elif col == "fin_semana":
                out[col] = out[col].clip(0, 1).round().astype(int)
    return out


def apply_rare_category_inflation(
    df: pd.DataFrame,
    columns: list[str],
    inflation_factor: float = 3.0,
    rare_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Incrementa la proporción de categorías raras (< rare_threshold)
    duplicando filas con esas categorías.
    """
    out = df.copy()
    extra_rows = []
    for col in columns:
        if col not in CATEGORICAL_FEATURES or col not in out.columns:
            continue
        freq = out[col].value_counts(normalize=True)
        rare = freq[freq < rare_threshold].index.tolist()
        if not rare:
            continue
        rare_mask = out[col].isin(rare)
        rare_df = out[rare_mask]
        n_dup = int(len(rare_df) * (inflation_factor - 1))
        if n_dup > 0:
            extra_rows.append(
                rare_df.sample(n=min(n_dup, len(rare_df)), replace=True, random_state=42)
            )
    if extra_rows:
        out = pd.concat([out] + extra_rows, ignore_index=True)
    return out


def apply_proportion_shift(
    df: pd.DataFrame,
    column: str,
    target_category: str,
    target_proportion: float,
) -> pd.DataFrame:
    """
    Altera proporciones categóricas re-muestreando hacia una categoría objetivo.
    """
    if column not in df.columns or column not in CATEGORICAL_FEATURES:
        return df.copy()

    out = df.copy()
    n = len(out)
    n_target = int(n * np.clip(target_proportion, 0.01, 0.99))
    n_rest = n - n_target

    other = out[out[column].astype(str) != str(target_category)]
    target_pool = out[out[column].astype(str) == str(target_category)]

    if len(other) == 0:
        other_sample = out.sample(n=n_rest, replace=True, random_state=42)
    else:
        other_sample = other.sample(n=min(n_rest, len(other)), replace=True, random_state=42)

    if len(target_pool) == 0:
        target_sample = out.copy()
        target_sample[column] = target_category
        target_sample = target_sample.sample(n=n_target, replace=True, random_state=42)
    else:
        target_sample = target_pool.sample(
            n=min(n_target, len(target_pool)), replace=True, random_state=42
        )

    # Completar si faltan filas
    while len(other_sample) < n_rest:
        other_sample = pd.concat(
            [other_sample, other.sample(min(n_rest - len(other_sample), len(other)), replace=True)],
            ignore_index=True,
        )
    while len(target_sample) < n_target:
        target_sample = pd.concat(
            [target_sample, target_pool.sample(min(n_target - len(target_sample), max(len(target_pool), 1)), replace=True)],
            ignore_index=True,
        )

    result = pd.concat(
        [other_sample.iloc[:n_rest], target_sample.iloc[:n_target]], ignore_index=True
    )
    return result.sample(frac=1, random_state=42).reset_index(drop=True)


def apply_all_alterations(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    mean_shift: float = 0.0,
    std_factor: float = 1.0,
    noise_sigma: float = 0.0,
    rare_inflation: float = 1.0,
    prop_column: str | None = None,
    prop_category: str | None = None,
    prop_target: float = 0.5,
) -> pd.DataFrame:
    """Aplica secuencialmente todas las alteraciones configuradas."""
    out = df.copy()
    if mean_shift != 0:
        out = apply_mean_shift(out, numeric_cols, mean_shift)
    if std_factor != 1.0:
        out = apply_std_inflation(out, numeric_cols, std_factor)
    if noise_sigma > 0:
        out = apply_noise_injection(out, numeric_cols, noise_sigma)
    if rare_inflation > 1.0:
        out = apply_rare_category_inflation(out, categorical_cols, rare_inflation)
    if prop_column and prop_category:
        out = apply_proportion_shift(out, prop_column, prop_category, prop_target)
    return out


def build_summary_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Tabla resumen before/after para numéricas y categóricas."""
    cols = columns or [c for c in FEATURE_COLUMNS if c in reference.columns]
    rows = []
    for col in cols:
        if col in NUMERIC_FEATURES:
            ref = reference[col].astype(float)
            cur = current[col].astype(float)
            rows.append(
                {
                    "Variable": col,
                    "Ref. Media": round(ref.mean(), 3),
                    "Alt. Media": round(cur.mean(), 3),
                    "Ref. Std": round(ref.std(), 3),
                    "Alt. Std": round(cur.std(), 3),
                    "Ref. Min": round(ref.min(), 3),
                    "Alt. Min": round(cur.min(), 3),
                    "Ref. Max": round(ref.max(), 3),
                    "Alt. Max": round(cur.max(), 3),
                }
            )
        else:
            ref_top = reference[col].value_counts().head(1)
            cur_top = current[col].value_counts().head(1)
            rows.append(
                {
                    "Variable": col,
                    "Ref. Media": ref_top.index[0] if len(ref_top) else "-",
                    "Alt. Media": cur_top.index[0] if len(cur_top) else "-",
                    "Ref. Std": f"{ref_top.iloc[0]/len(reference):.1%}" if len(ref_top) else "-",
                    "Alt. Std": f"{cur_top.iloc[0]/len(current):.1%}" if len(cur_top) else "-",
                    "Ref. Min": reference[col].nunique(),
                    "Alt. Min": current[col].nunique(),
                    "Ref. Max": "-",
                    "Alt. Max": "-",
                }
            )
    return pd.DataFrame(rows)
