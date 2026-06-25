"""
MaternIA - Entrenamiento integral materno + fetal
=================================================

Este script entrena dos modelos separados:
1. Modelo materno: Maternal Health Risk Dataset.
2. Modelo fetal: UCI Cardiotocography / Fetal Health Dataset.

No fusiona ambos datasets, porque no pertenecen a las mismas pacientes.
La integración se realiza en la app mediante una capa final de decisión de triaje.

Ejecución:
    python train_maternia_integral.py

Salidas:
    models/maternIA_maternal_risk_model.pkl
    models/maternIA_fetal_health_model.pkl
    reports/*
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MATERNAL_FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
MATERNAL_CLASS_ORDER = ["low risk", "mid risk", "high risk"]

FETAL_FEATURES = [
    "baseline_value",
    "accelerations",
    "fetal_movement",
    "uterine_contractions",
    "light_decelerations",
    "severe_decelerations",
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes",
    "histogram_mode",
    "histogram_mean",
    "histogram_median",
    "histogram_variance",
    "histogram_tendency",
]

FETAL_CLASS_NAMES = {1: "Normal", 2: "Sospechoso", 3: "Patológico"}

FETAL_ALIASES = {
    "LB": "baseline_value",
    "baseline value": "baseline_value",
    "baseline_value": "baseline_value",
    "AC": "accelerations",
    "accelerations": "accelerations",
    "FM": "fetal_movement",
    "fetal_movement": "fetal_movement",
    "fetal movement": "fetal_movement",
    "UC": "uterine_contractions",
    "uterine_contractions": "uterine_contractions",
    "uterine contractions": "uterine_contractions",
    "DL": "light_decelerations",
    "light_decelerations": "light_decelerations",
    "light decelerations": "light_decelerations",
    "DS": "severe_decelerations",
    "severe_decelerations": "severe_decelerations",
    "severe decelerations": "severe_decelerations",
    "DP": "prolongued_decelerations",
    "prolongued_decelerations": "prolongued_decelerations",
    "prolonged_decelerations": "prolongued_decelerations",
    "prolongued decelerations": "prolongued_decelerations",
    "ASTV": "abnormal_short_term_variability",
    "abnormal_short_term_variability": "abnormal_short_term_variability",
    "abnormal short term variability": "abnormal_short_term_variability",
    "MSTV": "mean_short_term_variability",
    "mean_short_term_variability": "mean_short_term_variability",
    "mean short term variability": "mean_short_term_variability",
    "ALTV": "percentage_of_time_with_abnormal_long_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability": "percentage_of_time_with_abnormal_long_term_variability",
    "percentage of time with abnormal long term variability": "percentage_of_time_with_abnormal_long_term_variability",
    "pct_abnormal_long_term_variability": "percentage_of_time_with_abnormal_long_term_variability",
    "MLTV": "mean_value_of_long_term_variability",
    "mean_value_of_long_term_variability": "mean_value_of_long_term_variability",
    "mean value of long term variability": "mean_value_of_long_term_variability",
    "mean_long_term_variability": "mean_value_of_long_term_variability",
    "Width": "histogram_width",
    "hist_width": "histogram_width",
    "histogram_width": "histogram_width",
    "Min": "histogram_min",
    "hist_min": "histogram_min",
    "histogram_min": "histogram_min",
    "Max": "histogram_max",
    "hist_max": "histogram_max",
    "histogram_max": "histogram_max",
    "Nmax": "histogram_number_of_peaks",
    "hist_n_peaks": "histogram_number_of_peaks",
    "histogram_number_of_peaks": "histogram_number_of_peaks",
    "Nzeros": "histogram_number_of_zeroes",
    "hist_n_zeros": "histogram_number_of_zeroes",
    "histogram_number_of_zeroes": "histogram_number_of_zeroes",
    "Mode": "histogram_mode",
    "hist_mode": "histogram_mode",
    "histogram_mode": "histogram_mode",
    "Mean": "histogram_mean",
    "hist_mean": "histogram_mean",
    "histogram_mean": "histogram_mean",
    "Median": "histogram_median",
    "hist_median": "histogram_median",
    "histogram_median": "histogram_median",
    "Variance": "histogram_variance",
    "hist_variance": "histogram_variance",
    "histogram_variance": "histogram_variance",
    "Tendency": "histogram_tendency",
    "hist_tendency": "histogram_tendency",
    "histogram_tendency": "histogram_tendency",
}


def normalize_risk_label(value: Any) -> str:
    text = str(value).strip().lower().replace("_", " ")
    if text in {"low", "low risk", "bajo", "riesgo bajo"}:
        return "low risk"
    if text in {"mid", "medium", "mid risk", "medium risk", "medio", "riesgo medio"}:
        return "mid risk"
    if text in {"high", "high risk", "alto", "riesgo alto"}:
        return "high risk"
    return text


def standardize_fetal_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        clean = str(col).strip()
        renamed[col] = FETAL_ALIASES.get(clean, FETAL_ALIASES.get(clean.replace(" ", "_"), clean))
    out = df.rename(columns=renamed).copy()
    return out


def safe_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def evaluate_predictions(y_train, y_pred_train, y_test, y_pred_test, positive_label, model_name: str) -> Dict[str, Any]:
    return {
        "Modelo": model_name,
        "Accuracy Train": round(float(accuracy_score(y_train, y_pred_train)), 4),
        "Accuracy Test": round(float(accuracy_score(y_test, y_pred_test)), 4),
        "Balanced Accuracy Train": round(float(balanced_accuracy_score(y_train, y_pred_train)), 4),
        "Balanced Accuracy Test": round(float(balanced_accuracy_score(y_test, y_pred_test)), 4),
        "Precision Macro Train": round(float(precision_score(y_train, y_pred_train, average="macro", zero_division=0)), 4),
        "Precision Macro Test": round(float(precision_score(y_test, y_pred_test, average="macro", zero_division=0)), 4),
        "Recall Macro Train": round(float(recall_score(y_train, y_pred_train, average="macro", zero_division=0)), 4),
        "Recall Macro Test": round(float(recall_score(y_test, y_pred_test, average="macro", zero_division=0)), 4),
        "F1 Macro Train": round(float(f1_score(y_train, y_pred_train, average="macro", zero_division=0)), 4),
        "F1 Macro Test": round(float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)), 4),
        "Recall Clase Crítica Train": round(float(recall_score(y_train, y_pred_train, labels=[positive_label], average="macro", zero_division=0)), 4),
        "Recall Clase Crítica Test": round(float(recall_score(y_test, y_pred_test, labels=[positive_label], average="macro", zero_division=0)), 4),
    }


def build_models(task: str) -> Dict[str, Pipeline]:
    models = {
        "LogisticRegression": Pipeline(steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, class_weight="balanced")),
        ]),
        "RandomForest": Pipeline(steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")),
        ]),
        "GradientBoosting": Pipeline(steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }
    if HAS_LIGHTGBM:
        models["LightGBM"] = Pipeline(steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                verbose=-1,
            )),
        ])
    return models


def plot_confusion(y_test, y_pred, labels: List[Any], display_labels: List[str], title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(model: Pipeline, feature_names: List[str], out_csv: Path, out_png: Path, title: str) -> None:
    clf = model.named_steps.get("clf")
    if hasattr(clf, "feature_importances_"):
        scores = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        scores = np.mean(np.abs(clf.coef_), axis=0)
    else:
        scores = np.zeros(len(feature_names))
    imp = pd.DataFrame({"variable": feature_names, "importancia": scores}).sort_values("importancia", ascending=False)
    imp.to_csv(out_csv, index=False)
    top = imp.head(15).sort_values("importancia", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["variable"], top["importancia"])
    ax.set_title(title)
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def load_maternal_data() -> Tuple[pd.DataFrame, pd.Series, str]:
    local_path = DATA_DIR / "maternal_health_risk.csv"
    if local_path.exists():
        df = pd.read_csv(local_path)
        source = str(local_path)
    else:
        print("No se encontró data/maternal_health_risk.csv. Descargando UCI Maternal Health Risk ID 863...")
        ds = fetch_ucirepo(id=863)
        X = ds.data.features.copy()
        ydf = ds.data.targets.copy()
        df = pd.concat([X, ydf], axis=1)
        source = "UCI Maternal Health Risk ID 863"

    missing = [c for c in MATERNAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas maternas: {missing}")
    target_col = "RiskLevel" if "RiskLevel" in df.columns else df.columns[-1]
    X = df[MATERNAL_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[target_col].apply(normalize_risk_label)
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True), source


def load_fetal_data() -> Tuple[pd.DataFrame, pd.Series, str]:
    local_path = DATA_DIR / "fetal_health.csv"
    if local_path.exists():
        df = pd.read_csv(local_path)
        source = str(local_path)
    else:
        print("No se encontró data/fetal_health.csv. Descargando UCI Cardiotocography ID 193...")
        ds = fetch_ucirepo(id=193)
        X = ds.data.features.copy()
        ydf = ds.data.targets.copy()
        df = pd.concat([X, ydf], axis=1)
        source = "UCI Cardiotocography ID 193"

    df = standardize_fetal_columns(df)
    missing = [c for c in FETAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas fetales CTG: {missing}")

    # UCI target NSP; Kaggle target fetal_health.
    if "NSP" in df.columns:
        target_col = "NSP"
    elif "fetal_health" in df.columns:
        target_col = "fetal_health"
    else:
        target_col = df.columns[-1]

    X = df[FETAL_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce").astype("Int64")
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask].reset_index(drop=True), y.loc[mask].astype(int).reset_index(drop=True), source


def train_task(
    X: pd.DataFrame,
    y: pd.Series,
    task_name: str,
    positive_label: Any,
    labels: List[Any],
    display_labels: List[str],
    model_path: Path,
    main_metric_name: str,
    class_names: Dict[Any, str],
    source: str,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"Entrenando tarea: {task_name}")
    print("Fuente:", source)
    print("Filas:", len(X), "| Columnas:", X.shape[1])
    print("Distribución de clases:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = build_models(task_name)
    rows = []
    predictions = {}
    fitted = {}

    for name, model in models.items():
        print(f"\nEntrenando: {name}")
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        metrics = evaluate_predictions(y_train, pred_train, y_test, pred_test, positive_label, name)
        rows.append(metrics)
        predictions[name] = pred_test
        fitted[name] = model
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # Ajuste rápido de LightGBM cuando está disponible.
    if HAS_LIGHTGBM:
        print("\nAjustando LightGBM con búsqueda ligera, priorizando clase crítica...")

        def critical_recall(y_true, y_pred):
            return recall_score(y_true, y_pred, labels=[positive_label], average="macro", zero_division=0)

        scorer = make_scorer(critical_recall)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        pipe = Pipeline(steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1)),
        ])
        params = {
            "clf__n_estimators": [150, 250, 350],
            "clf__learning_rate": [0.03, 0.05, 0.08],
            "clf__num_leaves": [15, 31, 45],
            "clf__max_depth": [-1, 4, 6],
            "clf__min_child_samples": [10, 20, 30],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }
        search = RandomizedSearchCV(
            pipe,
            param_distributions=params,
            n_iter=6,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        search.fit(X_train, y_train)
        tuned = search.best_estimator_
        pred_train = tuned.predict(X_train)
        pred_test = tuned.predict(X_test)
        metrics = evaluate_predictions(y_train, pred_train, y_test, pred_test, positive_label, "LightGBM_Tuned")
        metrics["Mejores parámetros"] = search.best_params_
        rows.append(metrics)
        predictions["LightGBM_Tuned"] = pred_test
        fitted["LightGBM_Tuned"] = tuned
        print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))

    df_metrics = pd.DataFrame(rows)
    df_ordered = df_metrics.sort_values(
        by=["Recall Clase Crítica Test", "F1 Macro Test", "Balanced Accuracy Test"],
        ascending=False,
    ).reset_index(drop=True)
    best_name = df_ordered.loc[0, "Modelo"]
    best_model = fitted[best_name]
    best_pred = predictions[best_name]

    prefix = "maternal" if task_name.lower().startswith("maternal") else "fetal"
    df_metrics.to_csv(REPORTS_DIR / f"{prefix}_metrics_report.csv", index=False)
    df_ordered.to_csv(REPORTS_DIR / f"{prefix}_metrics_report_ordered.csv", index=False)

    report_text = classification_report(y_test, best_pred, labels=labels, target_names=display_labels, zero_division=0)
    with open(REPORTS_DIR / f"{prefix}_classification_report_final.txt", "w", encoding="utf-8") as f:
        f.write(f"MaternIA - Reporte final {task_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Fuente: {source}\n")
        f.write(f"Modelo seleccionado: {best_name}\n")
        f.write(f"Métrica principal: {main_metric_name}\n\n")
        f.write(report_text)

    plot_confusion(
        y_test,
        best_pred,
        labels=labels,
        display_labels=display_labels,
        title=f"Matriz de confusión - {task_name} - {best_name}",
        out_path=REPORTS_DIR / f"{prefix}_confusion_matrix.png",
    )
    plot_feature_importance(
        best_model,
        list(X.columns),
        out_csv=REPORTS_DIR / f"{prefix}_feature_importance.csv",
        out_png=REPORTS_DIR / f"{prefix}_feature_importance.png",
        title=f"Variables importantes - {task_name}",
    )

    artifact = {
        "project": "MaternIA",
        "task": task_name,
        "model_name": best_name,
        "model": best_model,
        "feature_names": list(X.columns),
        "class_names": class_names,
        "labels": labels,
        "main_metric": main_metric_name,
        "source": source,
        "metrics": df_ordered.to_dict(orient="records"),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(artifact, model_path)
    print("\nModelo guardado en:", model_path)

    summary = {
        "task": task_name,
        "source": source,
        "rows": int(len(X)),
        "features": list(X.columns),
        "best_model": best_name,
        "main_metric": main_metric_name,
        "best_metrics": df_ordered.loc[0].to_dict(),
        "model_path": str(model_path),
    }
    safe_json(summary, REPORTS_DIR / f"{prefix}_training_summary.json")
    return summary


def generate_integrated_summary(maternal_summary: Dict[str, Any], fetal_summary: Dict[str, Any]) -> None:
    summary = {
        "project": "MaternIA - Sistema integral materno-fetal",
        "functional_datasets": [
            "Maternal Health Risk",
            "UCI Cardiotocography / Fetal Health Classification",
        ],
        "architecture": "Dos modelos separados integrados por una capa final de triaje. No se fusionan los datasets.",
        "maternal_model": maternal_summary,
        "fetal_model": fetal_summary,
        "integration_logic": [
            "Riesgo materno alto => prioridad alta o referencia según signos de alarma.",
            "Estado fetal patológico => referencia urgente.",
            "Estado fetal sospechoso + riesgo materno medio/alto => prioridad alta.",
            "Edad gestacional y signos clínicos se usan como contexto de seguridad.",
        ],
    }
    safe_json(summary, REPORTS_DIR / "integrated_training_summary.json")


def main():
    print("=" * 80)
    print("MaternIA - Entrenamiento integral materno + fetal")
    print("=" * 80)

    X_m, y_m, source_m = load_maternal_data()
    maternal_summary = train_task(
        X=X_m,
        y=y_m,
        task_name="Maternal Health Risk",
        positive_label="high risk",
        labels=MATERNAL_CLASS_ORDER,
        display_labels=["Bajo", "Medio", "Alto"],
        model_path=MODELS_DIR / "maternIA_maternal_risk_model.pkl",
        main_metric_name="Recall de Riesgo Alto",
        class_names={"low risk": "Bajo", "mid risk": "Medio", "high risk": "Alto"},
        source=source_m,
    )

    X_f, y_f, source_f = load_fetal_data()
    fetal_summary = train_task(
        X=X_f,
        y=y_f,
        task_name="Fetal Health / Cardiotocography",
        positive_label=3,
        labels=[1, 2, 3],
        display_labels=["Normal", "Sospechoso", "Patológico"],
        model_path=MODELS_DIR / "maternIA_fetal_health_model.pkl",
        main_metric_name="Recall Patológico",
        class_names=FETAL_CLASS_NAMES,
        source=source_f,
    )

    generate_integrated_summary(maternal_summary, fetal_summary)
    print("\nEntrenamiento integral finalizado.")
    print("Ejecuta la app con:")
    print("python -m streamlit run app_streamlit_maternia.py")


if __name__ == "__main__":
    main()
