"""
Aplicación Streamlit — Fase 3 Proyecto 3
Predicción de gravedad de accidentes de tránsito en Costa Rica.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.drift import (
    apply_all_alterations,
    build_summary_table,
    compute_drift_report,
)
from utils.prediction import (
    compute_shap_values,
    explain_prediction_text,
    get_feature_importances,
    load_artifacts,
    predict,
)
from utils.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    NUMERIC_FEATURES,
    compute_feature_defaults,
    compute_feature_ranges,
    get_categorical_options,
    load_dataset,
    load_reference_dataset,
    validate_inputs,
)
from utils.visualization import (
    plot_box_comparison,
    plot_categorical_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_histogram_comparison,
    plot_kde_comparison,
    plot_probability_gauge,
    plot_roc_curve,
    plot_shap_bar,
    render_traffic_light,
)

# ── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Accidentes CR — MLOps Fase 3",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #546E7A;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E86AB 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifacts():
    """Carga modelo y preprocessor una vez por sesión."""
    return load_artifacts()


@st.cache_data
def get_full_dataset() -> pd.DataFrame:
    try:
        return load_dataset()
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def get_reference_data() -> pd.DataFrame:
    try:
        return load_reference_dataset()
    except FileNotFoundError:
        return pd.DataFrame()


def init_session_state(full_df: pd.DataFrame):
    """Inicializa copia del dataset para el simulador de drift."""
    if "altered_df" not in st.session_state or st.session_state.altered_df.empty:
        st.session_state.altered_df = full_df.copy() if not full_df.empty else pd.DataFrame()
    if "drift_applied" not in st.session_state:
        st.session_state.drift_applied = False


def render_prediction_inputs(
    defaults: dict,
    ranges: dict,
    cat_options: dict,
) -> dict:
    """Renderiza controles UI para las 16 variables predictoras."""
    inputs = {}
    st.sidebar.markdown("### Variables del accidente")

    for col in FEATURE_COLUMNS:
        schema = FEATURE_SCHEMA.get(col, {})
        label = schema.get("label", col)
        ui_type = schema.get("ui", "dropdown")

        if ui_type == "slider" and col in ranges:
            r = ranges[col]
            val = defaults.get(col, r["min"])
            inputs[col] = st.sidebar.slider(
                label,
                min_value=r["min"],
                max_value=r["max"],
                value=float(val),
                step=r.get("step", 1.0),
                key=f"pred_{col}",
            )
        elif ui_type == "selectbox" and col == "fin_semana":
            inputs[col] = st.sidebar.selectbox(
                label,
                options=[0, 1],
                format_func=lambda x: "Sí" if x == 1 else "No",
                index=int(defaults.get(col, 0)),
                key=f"pred_{col}",
            )
        else:
            opts = cat_options.get(col, [])
            if not opts:
                opts = [str(defaults.get(col, ""))]
            default_val = defaults.get(col, opts[0])
            idx = opts.index(default_val) if default_val in opts else 0
            inputs[col] = st.sidebar.selectbox(
                label, options=opts, index=idx, key=f"pred_{col}"
            )

    return inputs


# ── Pestaña 1: Predicción ───────────────────────────────────────────────────────


def tab_prediction(artifacts, df: pd.DataFrame):
    st.markdown('<p class="main-header">Predicción de Gravedad</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Modifique las variables y observe la predicción en tiempo real.</p>',
        unsafe_allow_html=True,
    )

    if df.empty:
        st.error(
            "No se encontró `data/accidentes_cr_con_llm.csv`. "
            "Copie el dataset o ejecute `python scripts/train_model.py`."
        )
        return

    defaults = compute_feature_defaults(df)
    ranges = compute_feature_ranges(df)
    cat_options = get_categorical_options(df)

    inputs = render_prediction_inputs(defaults, ranges, cat_options)
    cleaned, warnings = validate_inputs(inputs, cat_options)
    for w in warnings:
        st.warning(w)

    if not artifacts.loaded:
        st.error(artifacts.message)
        st.info("La pestaña de drift sigue disponible sin modelo entrenado.")
        return

    # Predicción inmediata en cada rerun de Streamlit
    result = predict(cleaned, artifacts.model, artifacts.preprocessor)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        pred_color = "#C62828" if result["prediction"] == 1 else "#2E7D32"
        st.markdown(
            f"""<div class="metric-card" style="background:{pred_color};">
            <div style="font-size:0.9rem; opacity:0.9;">PREDICCIÓN</div>
            <div style="font-size:1.6rem; font-weight:700;">{result['label']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(explain_prediction_text(result))
        st.metric("Probabilidad clase grave", f"{result['probability_grave']:.1%}")
        st.metric("Probabilidad clase leve", f"{result['probability_leve']:.1%}")

    with col_right:
        st.plotly_chart(
            plot_probability_gauge(result["probability_grave"]),
            use_container_width=True,
            key="gauge_pred",
        )

    st.markdown("---")
    st.subheader("Importancia de variables")
    imp_df = get_feature_importances(artifacts.model, artifacts.preprocessor)
    if not imp_df.empty:
        st.plotly_chart(
            plot_feature_importance(imp_df),
            use_container_width=True,
            key="importance_pred",
        )

    st.subheader("Explicabilidad SHAP")
    compute_shap = st.checkbox(
        "Calcular explicación SHAP (puede tardar unos segundos)",
        value=False,
        key="shap_checkbox",
    )
    if compute_shap:
        try:
            ref = get_reference_data()
            bg = ref if not ref.empty else df.sample(min(100, len(df)), random_state=42)
            shap_result = compute_shap_values(
                cleaned,
                artifacts.model,
                artifacts.preprocessor,
                background=bg,
            )
            if shap_result:
                st.plotly_chart(
                    plot_shap_bar(shap_result),
                    use_container_width=True,
                    key="shap_chart",
                )
                st.caption(
                    f"Valor base del modelo: {shap_result['base_value']:.4f}. "
                    "Valores positivos empujan hacia clase grave."
                )
            else:
                st.info("Instale `shap` (`pip install shap`) para ver explicaciones SHAP.")
        except Exception as exc:
            st.warning(f"No se pudo calcular SHAP: {exc}")

    # Advertencia de desbalance (del metadata si existe)
    if artifacts.metadata:
        recall = artifacts.metadata.get("metrics", {}).get("recall", 0)
        if recall < 0.35:
            st.info(
                f"Nota: el modelo tiene recall histórico de {recall:.1%} en prueba. "
                "Una probabilidad baja no garantiza ausencia de casos graves."
            )


# ── Pestaña 2: Simulador de Drift ───────────────────────────────────────────────


def tab_drift(full_df: pd.DataFrame, ref_df: pd.DataFrame):
    st.markdown('<p class="main-header">Simulador de Drift</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Modo auditor: altere distribuciones globalmente y evalúe el drift en tiempo real.</p>',
        unsafe_allow_html=True,
    )

    if full_df.empty:
        st.error("Dataset no disponible para simulación de drift.")
        return

    if ref_df.empty:
        st.warning(
            "No hay `data/referencia.csv`. Se usa muestra del dataset como referencia. "
            "Ejecute `python scripts/train_model.py` para la referencia oficial."
        )
        ref_df = full_df.sample(frac=0.8, random_state=42)

    init_session_state(full_df)

    # ── Controles del villano (visibles en UI) ────────────────────────────────
    with st.expander("Controles de alteración global (modo auditor)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Mean Shift**")
            mean_shift = st.slider(
                "Desplazamiento de medias (Δ)",
                -50.0,
                50.0,
                0.0,
                0.5,
                key="drift_mean_shift",
                help="Suma Δ a cada variable numérica seleccionada.",
            )
            st.markdown("**Std Deviation Shift**")
            std_factor = st.slider(
                "Factor de desviación estándar",
                0.5,
                3.0,
                1.0,
                0.1,
                key="drift_std_factor",
                help="x' = media + (x - media) × factor",
            )
        with c2:
            st.markdown("**Noise Injection**")
            noise_sigma = st.slider(
                "Sigma de ruido gaussiano",
                0.0,
                20.0,
                0.0,
                0.5,
                key="drift_noise",
                help="Inyecta N(0, σ) en variables numéricas.",
            )
            st.markdown("**Rare Category Inflation**")
            rare_inflation = st.slider(
                "Factor inflación categorías raras",
                1.0,
                5.0,
                1.0,
                0.1,
                key="drift_rare",
                help="Duplica filas con categorías de frecuencia < 2%.",
            )
        with c3:
            num_cols_sel = st.multiselect(
                "Variables numéricas a alterar",
                NUMERIC_FEATURES,
                default=NUMERIC_FEATURES,
                key="drift_num_cols",
            )
            cat_cols_sel = st.multiselect(
                "Variables categóricas (rare inflation)",
                CATEGORICAL_FEATURES,
                default=CATEGORICAL_FEATURES[:3],
                key="drift_cat_cols",
            )

        st.markdown("**Proportion Shift**")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            prop_col = st.selectbox(
                "Variable categórica",
                ["(ninguna)"] + CATEGORICAL_FEATURES,
                key="drift_prop_col",
            )
        with pc2:
            prop_cat = ""
            if prop_col != "(ninguna)":
                opts = full_df[prop_col].value_counts().head(15).index.tolist()
                prop_cat = st.selectbox("Categoría objetivo", opts, key="drift_prop_cat")
        with pc3:
            prop_target = (
                st.slider("Proporción objetivo (%)", 5, 95, 50, 5, key="drift_prop_pct")
                / 100.0
            )

        b1, b2 = st.columns(2)
        with b1:
            apply_btn = st.button("Aplicar alteraciones", type="primary", key="drift_apply")
        with b2:
            reset_btn = st.button("Restaurar dataset original", key="drift_reset")

        if apply_btn:
            st.session_state.altered_df = apply_all_alterations(
                full_df,
                numeric_cols=num_cols_sel or NUMERIC_FEATURES,
                categorical_cols=cat_cols_sel or CATEGORICAL_FEATURES,
                mean_shift=mean_shift,
                std_factor=std_factor,
                noise_sigma=noise_sigma,
                rare_inflation=rare_inflation,
                prop_column=prop_col if prop_col != "(ninguna)" else None,
                prop_category=prop_cat if prop_col != "(ninguna)" else None,
                prop_target=prop_target,
            )
            st.session_state.drift_applied = True
            st.rerun()

        if reset_btn:
            st.session_state.altered_df = full_df.copy()
            st.session_state.drift_applied = False
            st.rerun()

    altered = st.session_state.altered_df
    drift_report = compute_drift_report(ref_df, altered)

    # ── Semáforo global (dinámico según PSI) ──────────────────────────────────
    st.markdown("### Semáforo de riesgo global")
    st.markdown(
        render_traffic_light(
            drift_report["global_status_color"],
            drift_report["global_status_label"],
            f"{drift_report['global_status_message']} — "
            f"Drift Score global (PSI ponderado): {drift_report['global_psi']:.4f}",
        ),
        unsafe_allow_html=True,
    )

    # ── Métricas visibles: PSI, KS, Δ Media, Δ Std ─────────────────────────────
    st.subheader("Métricas de drift por variable")
    drift_table = pd.DataFrame(
        [
            {
                "Variable": r["feature"],
                "PSI": round(r["psi"], 4),
                "KS": round(r["ks"], 4),
                "Δ Media (norm.)": round(r["delta_mean"], 4),
                "Δ Std (norm.)": round(r["delta_std"], 4),
                "Semáforo": r["status_label"],
                "Mensaje": r["status_message"],
            }
            for r in drift_report["features"]
        ]
    )
    st.dataframe(drift_table, use_container_width=True, hide_index=True)

    # Semáforos por feature (dinámicos)
    st.subheader("Semáforo por variable")
    sem_cols = st.columns(4)
    for i, row in enumerate(drift_report["features"]):
        with sem_cols[i % 4]:
            st.markdown(
                render_traffic_light(
                    row["status_color"],
                    f"{row['feature'][:12]}…" if len(row["feature"]) > 12 else row["feature"],
                    f"PSI={row['psi']:.3f}",
                ),
                unsafe_allow_html=True,
            )

    # ── Visualizaciones before/after ───────────────────────────────────────────
    st.subheader("Comparación visual: Referencia vs Alterado")
    viz_col = st.selectbox(
        "Variable a visualizar",
        FEATURE_COLUMNS,
        key="drift_viz_col",
    )

    vc1, vc2 = st.columns(2)
    with vc1:
        if viz_col in NUMERIC_FEATURES:
            st.plotly_chart(
                plot_histogram_comparison(
                    ref_df[viz_col], altered[viz_col], f"Histograma — {viz_col}"
                ),
                use_container_width=True,
                key="drift_hist",
            )
        else:
            st.plotly_chart(
                plot_categorical_comparison(
                    ref_df[viz_col], altered[viz_col], f"Frecuencias — {viz_col}"
                ),
                use_container_width=True,
                key="drift_cat",
            )
    with vc2:
        if viz_col in NUMERIC_FEATURES:
            st.plotly_chart(
                plot_kde_comparison(ref_df[viz_col], altered[viz_col], f"KDE — {viz_col}"),
                use_container_width=True,
                key="drift_kde",
            )
            st.plotly_chart(
                plot_box_comparison(ref_df[viz_col], altered[viz_col], f"Boxplot — {viz_col}"),
                use_container_width=True,
                key="drift_box",
            )
        else:
            st.info("KDE y boxplot aplican solo a variables numéricas.")

    st.subheader("Tabla resumen antes / después")
    summary = build_summary_table(ref_df, altered)
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ── Pestaña 3: Información del Modelo ─────────────────────────────────────────


def tab_model_info(artifacts):
    st.markdown('<p class="main-header">Información del Modelo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Métricas, configuración e interpretabilidad del modelo en producción.</p>',
        unsafe_allow_html=True,
    )

    if not artifacts.loaded or not artifacts.metadata:
        st.error(artifacts.message)
        st.markdown(
            """
            **Pasos para generar artefactos:**
            ```bash
            pip install -r requirements.txt
            python scripts/train_model.py
            ```
            """
        )
        return

    meta = artifacts.metadata
    metrics = meta.get("metrics", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    c2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    c3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    c4.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
    c5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")

    st.markdown("---")
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("#### Configuración")
        st.markdown(f"- **Modelo:** {meta.get('model_name', 'N/A')}")
        st.markdown(f"- **Criterio de selección:** {meta.get('selection_criterion', 'N/A')}")
        st.markdown(f"- **Variables predictoras:** {meta.get('n_features', 16)}")
        st.markdown(f"- **Tamaño entrenamiento:** {meta.get('train_size', 'N/A'):,}")
        st.markdown(f"- **Tamaño prueba:** {meta.get('test_set_size', 'N/A'):,}")
        st.markdown(f"- **Fecha de entrenamiento:** {meta.get('training_date', 'N/A')}")
        st.markdown(f"- **Random state:** {meta.get('random_state', 42)}")

        hp = meta.get("best_hyperparameters", {})
        if hp:
            st.markdown("**Hiperparámetros ganadores (GridSearch):**")
            for k, v in hp.items():
                st.markdown(f"  - `{k}`: {v}")

    with ic2:
        st.markdown("#### Preprocesamiento")
        prep = meta.get("preprocessing", {})
        st.markdown(f"- Numéricas: `{prep.get('numeric', 'StandardScaler')}`")
        st.markdown(f"- Categóricas: `{prep.get('categorical', 'TargetEncoder')}`")
        st.markdown(f"- **Target:** `{meta.get('target_column', 'clase_bin')}`")
        interp = meta.get("target_interpretation", {})
        st.markdown(f"  - 0: {interp.get('0', 'Leve')}")
        st.markdown(f"  - 1: {interp.get('1', 'Grave')}")

    st.markdown("---")
    gc1, gc2 = st.columns(2)
    with gc1:
        cm = meta.get("confusion_matrix", [[0, 0], [0, 0]])
        st.plotly_chart(
            plot_confusion_matrix(cm),
            use_container_width=True,
            key="cm_model_info",
        )
    with gc2:
        roc = meta.get("roc_curve", {})
        if roc:
            st.plotly_chart(
                plot_roc_curve(roc["fpr"], roc["tpr"], metrics.get("roc_auc", 0)),
                use_container_width=True,
                key="roc_model_info",
            )

    imp = meta.get("feature_importances", {})
    if imp:
        imp_df = pd.DataFrame(
            {"feature": list(imp.keys()), "importance": list(imp.values())}
        ).sort_values("importance", ascending=False)
        st.subheader("Importancia de variables")
        st.plotly_chart(
            plot_feature_importance(imp_df),
            use_container_width=True,
            key="importance_model_info",
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    st.sidebar.markdown("## Accidentes CR")
    st.sidebar.markdown("*Proyecto 3 — Fase 3 MLOps*")
    st.sidebar.markdown("---")

    artifacts = get_artifacts()
    full_df = get_full_dataset()
    ref_df = get_reference_data()

    if artifacts.loaded:
        st.sidebar.success("Modelo cargado")
    else:
        st.sidebar.error("Modelo no disponible")
        st.sidebar.caption(artifacts.message)

    tab1, tab2, tab3 = st.tabs(
        ["Predicción", "Simulador de Drift", "Info del Modelo"]
    )

    with tab1:
        tab_prediction(artifacts, full_df)
    with tab2:
        tab_drift(full_df, ref_df)
    with tab3:
        tab_model_info(artifacts)


if __name__ == "__main__":
    main()
