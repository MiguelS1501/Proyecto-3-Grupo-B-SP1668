"""
Visualizaciones Plotly reutilizables para predicción, drift e info del modelo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Paleta corporativa
COLORS = {
    "primary": "#1E3A5F",
    "secondary": "#2E86AB",
    "accent": "#F18F01",
    "danger": "#C62828",
    "success": "#2E7D32",
    "warning": "#F9A825",
    "muted": "#78909C",
    "ref": "#2E86AB",
    "alt": "#F18F01",
}


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Aplica estilo consistente a figuras Plotly."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=13),
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#FAFBFC",
    )
    return fig


def plot_probability_gauge(prob_grave: float, label: str = "Prob. accidente grave") -> go.Figure:
    """Gauge de probabilidad de clase positiva."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_grave * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": label, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLORS["danger"] if prob_grave > 0.5 else COLORS["success"]},
                "steps": [
                    {"range": [0, 30], "color": "#E8F5E9"},
                    {"range": [30, 60], "color": "#FFF8E1"},
                    {"range": [60, 100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": COLORS["primary"], "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    return apply_plotly_theme(fig)


def plot_feature_importance(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Bar chart horizontal de importancias."""
    data = df.head(top_n).sort_values("importance")
    fig = px.bar(
        data,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        title="Importancia de variables (Random Forest)",
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return apply_plotly_theme(fig)


def plot_shap_bar(shap_result: dict[str, Any], top_n: int = 12) -> go.Figure:
    """Bar chart de valores SHAP."""
    features = shap_result["features"]
    # Asegurar escalares float (evita abs() sobre listas anidadas de SHAP)
    values = [float(np.ravel(v)[0]) for v in shap_result["shap_values"]]
    df = pd.DataFrame({"feature": features, "shap": values})
    df["abs_shap"] = df["shap"].astype(float).abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n).sort_values("shap")

    colors = [COLORS["danger"] if float(v) > 0 else COLORS["secondary"] for v in df["shap"]]
    fig = go.Figure(
        go.Bar(
            x=df["shap"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in df["shap"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Contribución SHAP a clase grave (+ aumenta riesgo)",
        xaxis_title="Valor SHAP",
        yaxis_title="",
    )
    return apply_plotly_theme(fig)


def plot_histogram_comparison(
    reference: pd.Series,
    current: pd.Series,
    title: str,
) -> go.Figure:
    """Histograma superpuesto referencia vs alterado."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=reference.astype(float),
            name="Referencia",
            opacity=0.6,
            marker_color=COLORS["ref"],
            nbinsx=30,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=current.astype(float),
            name="Alterado",
            opacity=0.6,
            marker_color=COLORS["alt"],
            nbinsx=30,
        )
    )
    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis_title=reference.name,
        yaxis_title="Frecuencia",
        legend=dict(orientation="h", y=1.12),
    )
    return apply_plotly_theme(fig)


def plot_box_comparison(
    reference: pd.Series,
    current: pd.Series,
    title: str,
) -> go.Figure:
    """Boxplot lado a lado."""
    df = pd.DataFrame(
        {
            "valor": pd.concat(
                [reference.astype(float), current.astype(float)], ignore_index=True
            ),
            "dataset": ["Referencia"] * len(reference) + ["Alterado"] * len(current),
        }
    )
    fig = px.box(
        df,
        x="dataset",
        y="valor",
        color="dataset",
        color_discrete_map={"Referencia": COLORS["ref"], "Alterado": COLORS["alt"]},
        title=title,
    )
    fig.update_layout(showlegend=False)
    return apply_plotly_theme(fig)


def plot_kde_comparison(
    reference: pd.Series,
    current: pd.Series,
    title: str,
) -> go.Figure:
    """Curvas de densidad (KDE) comparativas."""
    fig = go.Figure()
    for series, name, color in [
        (reference.astype(float), "Referencia", COLORS["ref"]),
        (current.astype(float), "Alterado", COLORS["alt"]),
    ]:
        data = series.dropna()
        if len(data) < 2:
            continue
        kde_x = np.linspace(data.min(), data.max(), 200)
        # KDE gaussiana simple
        bandwidth = data.std() * (len(data) ** -0.2) or 1.0
        density = np.mean(
            [
                np.exp(-0.5 * ((kde_x - xi) / bandwidth) ** 2)
                / (bandwidth * np.sqrt(2 * np.pi))
                for xi in data.sample(min(500, len(data)), random_state=42)
            ],
            axis=0,
        )
        fig.add_trace(
            go.Scatter(x=kde_x, y=density, mode="lines", name=name, line=dict(color=color, width=2))
        )
    fig.update_layout(
        title=title,
        xaxis_title=reference.name,
        yaxis_title="Densidad",
        legend=dict(orientation="h", y=1.12),
    )
    return apply_plotly_theme(fig)


def plot_categorical_comparison(
    reference: pd.Series,
    current: pd.Series,
    title: str,
    top_n: int = 10,
) -> go.Figure:
    """Barras de frecuencia para categóricas."""
    ref_pct = reference.value_counts(normalize=True).head(top_n)
    cur_pct = current.value_counts(normalize=True).reindex(ref_pct.index, fill_value=0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Referencia",
            x=ref_pct.index.astype(str),
            y=ref_pct.values,
            marker_color=COLORS["ref"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Alterado",
            x=cur_pct.index.astype(str),
            y=cur_pct.values,
            marker_color=COLORS["alt"],
        )
    )
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title=reference.name,
        yaxis_title="Proporción",
        legend=dict(orientation="h", y=1.12),
    )
    return apply_plotly_theme(fig)


def plot_drift_table_chart(drift_rows: list[dict]) -> go.Figure:
    """Tabla visual de métricas de drift con colores de semáforo."""
    color_map = {"green": "#C8E6C9", "yellow": "#FFF9C4", "red": "#FFCDD2"}
    df = pd.DataFrame(drift_rows)
    cell_colors = [
        [color_map.get(c, "white") for c in df["status_color"]]
        for _ in range(len(df.columns))
    ]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns),
                    fill_color=COLORS["primary"],
                    font=dict(color="white", size=12),
                    align="center",
                ),
                cells=dict(
                    values=[df[c] for c in df.columns],
                    fill_color=cell_colors,
                    align="center",
                    font=dict(size=11),
                ),
            )
        ]
    )
    fig.update_layout(title="Métricas de drift por variable", height=400 + len(df) * 25)
    return apply_plotly_theme(fig)


def plot_confusion_matrix(cm: list[list[int]], labels: list[str] | None = None) -> go.Figure:
    """Heatmap de matriz de confusión."""
    labels = labels or ["Leve (0)", "Grave (1)"]
    cm_arr = np.array(cm)
    fig = px.imshow(
        cm_arr,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title="Matriz de confusión (conjunto de prueba)",
    )
    fig.update_layout(xaxis_title="Predicho", yaxis_title="Real")
    return apply_plotly_theme(fig)


def plot_roc_curve(fpr: list, tpr: list, auc: float) -> go.Figure:
    """Curva ROC."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC={auc:.4f})",
            line=dict(color=COLORS["secondary"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Azar",
            line=dict(color=COLORS["muted"], dash="dash"),
        )
    )
    fig.update_layout(
        title="Curva ROC",
        xaxis_title="Tasa de falsos positivos",
        yaxis_title="Tasa de verdaderos positivos",
    )
    return apply_plotly_theme(fig)


def render_traffic_light(color: str, label: str, message: str) -> str:
    """HTML para semáforo de riesgo."""
    color_hex = {
        "green": COLORS["success"],
        "yellow": COLORS["warning"],
        "red": COLORS["danger"],
    }.get(color, COLORS["muted"])
    return f"""
    <div style="
        border-left: 6px solid {color_hex};
        background: #F5F7FA;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 8px 0;
    ">
        <span style="
            background:{color_hex};
            color:white;
            padding:4px 12px;
            border-radius:20px;
            font-weight:700;
            font-size:13px;
        ">{label}</span>
        <p style="margin:10px 0 0 0; color:#37474F; font-size:15px;">{message}</p>
    </div>
    """
