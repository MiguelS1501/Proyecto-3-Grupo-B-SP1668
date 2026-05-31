# Plan de Implementación — Fase 3 (Proyecto 3)

Documento de referencia generado a partir del análisis de `proyecto__3_grupo_b.py` y `accidentes_cr_con_llm.csv`.

## Modelo de producción

- **Algoritmo:** Random Forest (Con LLM) — `best_rf2`
- **Target:** `clase_bin` (1 = muertos/graves, 0 = solo leves)
- **Features:** 16 predictoras (4 numéricas + 12 categóricas)
- **Preprocesamiento:** StandardScaler + TargetEncoder
- **Criterio de selección:** F1-Score (0.4225, ROC-AUC 0.7376)
- **Ensamblaje descartado:** VotingClassifier (mayor Accuracy, peor F1)

## Arquitectura implementada

Ver `README.md` y módulos en `utils/` para detalle de PSI, KS, SHAP y UI Streamlit.

## Ejecución

```bash
pip install -r requirements.txt
python scripts/train_model.py
streamlit run app.py
```

