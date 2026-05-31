# Proyecto 3 — Fase 3: Aplicación MLOps con Streamlit

Aplicación interactiva para predicción de gravedad de accidentes de tránsito en Costa Rica, simulación de drift y panel de métricas del modelo **Random Forest (Con LLM)**.

## Requisitos

- Python 3.10+
- Dataset `accidentes_cr_con_llm.csv` (incluido en `data/` o en el directorio padre)

## Instalación

```bash
cd Aplicación_Interactiva_Proyecto_3
pip install -r requirements.txt
```

## Entrenamiento del modelo (obligatorio antes del primer uso)

El notebook original no exportó artefactos `.pkl`. Ejecute el script de entrenamiento reproducible:

```bash
python scripts/train_model.py
```

Esto genera:

- `models/modelo_final.pkl` — Random Forest optimizado
- `models/preprocessor.pkl` — ColumnTransformer (StandardScaler + TargetEncoder)
- `models/scaler.pkl` — StandardScaler extraído
- `models/model_metadata.json` — métricas, matriz de confusión, importancias
- `data/referencia.csv` — subconjunto de entrenamiento para drift

## Ejecutar la aplicación

```bash
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`.

## Estructura del proyecto

```
Aplicación_Interactiva_Proyecto_3/
├── app.py
├── models/
│   ├── modelo_final.pkl
│   ├── preprocessor.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── data/
│   ├── accidentes_cr_con_llm.csv
│   └── referencia.csv
├── utils/
│   ├── drift.py
│   ├── prediction.py
│   ├── preprocessing.py
│   └── visualization.py
├── scripts/
│   └── train_model.py
├── requirements.txt
└── README.md
```

## Pestañas de la aplicación

### 1. Predicción
- 16 controles interactivos (sliders, dropdowns, selectbox)
- Predicción y probabilidad en tiempo real
- Importancia de variables y SHAP (opcional)

### 2. Simulador de Drift
- Alteración global de distribuciones (medias, std, ruido, categorías raras, proporciones)
- Métricas: PSI, KS, Δ media, Δ std, Drift Score global
- Semáforo VERDE / AMARILLO / ROJO
- Visualizaciones: histogramas, boxplots, KDE, tablas resumen

### 3. Info del Modelo
- Métricas de evaluación (Accuracy, Precision, Recall, F1, ROC-AUC)
- Matriz de confusión, curva ROC, importancias
- Hiperparámetros y fecha de entrenamiento

## Modelo seleccionado

| Atributo | Valor |
|---|---|
| Algoritmo | Random Forest (Con LLM) |
| Target | `clase_bin` (1 = muertos/graves) |
| Criterio | F1-Score |
| Features | 16 (incluye `severidad_tipo_LLM`, `cobertura_emergencias_canton`) |
| Preprocesamiento | StandardScaler + TargetEncoder |

## Validación

Tras entrenar, verifique que F1-Score ≈ 0.4225 (±0.005) con `random_state=42`.

## Robustez

- Si faltan artefactos, la app muestra mensajes claros con instrucciones
- El simulador de drift funciona sin modelo cargado
- Las alteraciones de drift no modifican el CSV original

## Autores

Grupo B — Aprendizaje de Máquina, UCR
