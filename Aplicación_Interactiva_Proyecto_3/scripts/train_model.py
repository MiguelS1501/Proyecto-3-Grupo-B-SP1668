"""
Entrena el modelo Random Forest (Con LLM) y exporta artefactos para producción.
Replica la lógica de proyecto__3_grupo_b.py (líneas 880-964).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Permitir importar utils desde la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    DATA_DIR,
    MODELS_DIR,
    get_data_path,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construye el ColumnTransformer idéntico al notebook."""
    numeric_features = list(NUMERIC_FEATURES)
    categorical_features = list(CATEGORICAL_FEATURES)

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("target_enc", TargetEncoder())])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_model(model, X_test, y_test) -> dict:
    """Calcula métricas de evaluación en el conjunto de prueba."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }


def main() -> None:
    """Pipeline completo de entrenamiento y exportación."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data_path = get_data_path()
    if not data_path.exists():
        # Intentar copiar desde el directorio padre
        parent_csv = PROJECT_ROOT.parent / "accidentes_cr_con_llm.csv"
        if parent_csv.exists():
            import shutil

            shutil.copy2(parent_csv, data_path)
            print(f"Dataset copiado desde {parent_csv}")
        else:
            raise FileNotFoundError(
                f"No se encontró {data_path}. Coloque accidentes_cr_con_llm.csv en data/"
            )

    print(f"Cargando dataset: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Entrenamiento: {X_train.shape[0]:,} | Prueba: {X_test.shape[0]:,}")

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t = preprocessor.transform(X_test)

    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [5, 10],
    }
    rf = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_cv = GridSearchCV(
        estimator=rf,
        param_grid=rf_params,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )

    print("Entrenando Random Forest (GridSearchCV, scoring=f1)...")
    rf_cv.fit(X_train_t, y_train)
    best_model = rf_cv.best_estimator_

    metrics = evaluate_model(best_model, X_test_t, y_test)
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"Mejores params: {rf_cv.best_params_}")

    # Importancias alineadas con nombres de features transformadas
    feature_names = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    importances = dict(
        zip(feature_names, best_model.feature_importances_.tolist())
    )
    importances_sorted = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )

    # Extraer scaler del preprocessor
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    # Exportar artefactos
    joblib.dump(best_model, MODELS_DIR / "modelo_final.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # Referencia para drift (solo features, split de entrenamiento)
    ref_path = DATA_DIR / "referencia.csv"
    X_train.to_csv(ref_path, index=False)
    print(f"Referencia guardada: {ref_path} ({X_train.shape[0]:,} filas)")

    metadata = {
        "model_name": "Random Forest (Con LLM)",
        "algorithm": "RandomForestClassifier",
        "selection_criterion": "F1-Score",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_size": int(X_train.shape[0]),
        "test_set_size": int(X_test.shape[0]),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "target_interpretation": {
            "0": "Solo leves",
            "1": "Con muertos o heridos graves",
        },
        "best_hyperparameters": rf_cv.best_params_,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "roc_curve": metrics["roc_curve"],
        "feature_importances": importances_sorted,
        "class_weight": "balanced",
        "preprocessing": {
            "numeric": "StandardScaler",
            "categorical": "TargetEncoder",
        },
    }

    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Modelo guardado en {MODELS_DIR}")
    print(f"Metadata guardada en {meta_path}")
    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()
