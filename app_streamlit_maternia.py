# ================================================================
# MaternIA Integral - Interfaz hospitalaria
# Reemplaza tu archivo: app_streamlit_maternia.py
# Ejecuta con: python -m streamlit run app_streamlit_maternia.py
# ================================================================

import os
import re
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.cognitive_triage_service import cognitive_triage_service

# ------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------
st.set_page_config(
    page_title="MaternIA Integral",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

MATERNAL_MODEL_PATHS = [
    MODELS_DIR / "maternIA_maternal_risk_model.pkl",
    MODELS_DIR / "maternIA_maternal_risk.pkl",
    BASE_DIR / "maternIA_maternal_risk_model.pkl",
    BASE_DIR / "maternIA_maternal_risk.pkl",
]

FETAL_MODEL_PATHS = [
    MODELS_DIR / "maternIA_fetal_health_model.pkl",
    MODELS_DIR / "maternIA_fetal_health.pkl",
    BASE_DIR / "maternIA_fetal_health_model.pkl",
    BASE_DIR / "maternIA_fetal_health.pkl",
]

FETAL_MLP_MODEL_PATHS = [
    MODELS_DIR / "maternIA_fetal_mlp_neural_network.pkl",
    BASE_DIR / "maternIA_fetal_mlp_neural_network.pkl",
]

DEFAULT_MATERNAL_FEATURES = [
    "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"
]

DEFAULT_FETAL_FEATURES_UCI = [
    "LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV", "MLTV",
    "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean", "Median", "Variance", "Tendency"
]

# ------------------------------------------------
# ESTILO VISUAL - HOSPITAL REAL
# ------------------------------------------------
st.markdown(
    """
    <style>
    /* =======================================================
       MaternIA - Diseño hospitalario consistente y legible
       Forzamos tema claro en el contenido principal.
       ======================================================= */
    :root {
        --bg: #f4f8fb;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --primary: #087f8c;
        --primary-2: #0ea5e9;
        --primary-dark: #075985;
        --text: #0f172a;
        --text-soft: #475569;
        --muted: #64748b;
        --border: #dbe5ee;
        --success: #16a34a;
        --warning: #d97706;
        --danger: #dc2626;
        --sidebar-bg: #0b1220;
    }

    /* Fondo principal */
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #eef7f9 100%) !important;
        color: var(--text) !important;
    }

    .block-container {
        padding-top: 2rem !important;
        max-width: 1240px !important;
    }

    /* Texto del contenido principal: siempre oscuro */
    main, main p, main div, main span, main label,
    main h1, main h2, main h3, main h4, main h5, main h6,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] * {
        color: var(--text) !important;
    }

    main p, main span, main label {
        font-size: 0.96rem;
    }

    /* Sidebar con contraste estable */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071226 0%, #111827 100%) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent !important;
    }

    /* Encabezado */
    .main-header {
        padding: 1.55rem 1.75rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #087f8c 0%, #0284c7 100%);
        box-shadow: 0 18px 42px rgba(2, 132, 199, 0.20);
        margin-bottom: 1.2rem;
    }
    .main-header, .main-header * {
        color: #ffffff !important;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.25rem;
        font-weight: 900;
        letter-spacing: -0.035em;
    }
    .main-header p {
        margin: 0.45rem 0 0 0;
        font-size: 1.02rem;
        font-weight: 500;
        opacity: 0.98;
    }

    /* Banners y tarjetas */
    .safe-banner {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-left: 8px solid #f97316;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
        font-weight: 750;
    }
    .safe-banner, .safe-banner * { color: #7c2d12 !important; }

    .status-ok, .status-missing, .card, .result-card, [data-testid="stMetric"] {
        box-sizing: border-box;
    }

    .status-ok {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-left: 8px solid #22c55e;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        font-weight: 800;
    }
    .status-ok, .status-ok * { color: #14532d !important; }

    .status-missing {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 8px solid #ef4444;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        font-weight: 800;
    }
    .status-missing, .status-missing * { color: #7f1d1d !important; }

    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        margin-bottom: 1.1rem;
    }
    .card-title {
        font-weight: 900;
        font-size: 1.14rem;
        margin-bottom: 0.45rem;
        color: var(--text) !important;
    }
    .card-subtitle {
        color: var(--text-soft) !important;
        font-size: 0.94rem;
        margin-bottom: 0.7rem;
    }

    /* Widgets: etiquetas legibles */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stFileUploader"] label,
    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {
        color: var(--text) !important;
        font-weight: 800 !important;
    }

    /* Inputs numéricos */
    [data-testid="stNumberInput"] input,
    [data-baseweb="input"] input,
    input[type="number"],
    input[type="text"],
    textarea {
        background: #ffffff !important;
        color: #111827 !important;
        border-color: #cbd5e1 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    [data-testid="stNumberInput"] input::placeholder,
    input::placeholder,
    textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }
    [data-testid="stNumberInput"] div,
    [data-baseweb="input"] {
        color: #111827 !important;
    }
    [data-testid="stNumberInput"] button {
        background: #f1f5f9 !important;
        color: #0f172a !important;
        border-color: #cbd5e1 !important;
    }

    /* Selectbox */
    [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #111827 !important;
        border-color: #cbd5e1 !important;
        font-weight: 700 !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        color: #111827 !important;
    }

    /* Checkboxes y radios en el contenido principal */
    main [data-testid="stCheckbox"] label,
    main [data-testid="stCheckbox"] label *,
    main [data-testid="stRadio"] label,
    main [data-testid="stRadio"] label * {
        color: var(--text) !important;
        font-weight: 700 !important;
    }
    main [data-testid="stCheckbox"] svg,
    main [data-testid="stRadio"] svg {
        color: var(--primary) !important;
        fill: var(--primary) !important;
    }

    /* Botones */
    .stButton > button,
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(90deg, #087f8c, #0284c7) !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.15rem !important;
        font-weight: 900 !important;
        box-shadow: 0 8px 18px rgba(2, 132, 199, 0.25) !important;
    }
    .stButton > button *,
    [data-testid="stDownloadButton"] button * {
        color: #ffffff !important;
    }
    .stButton > button:hover,
    [data-testid="stDownloadButton"] button:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
    }

    /* Resultados */
    .result-card {
        border-radius: 22px;
        padding: 1.35rem;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.10);
        margin-bottom: 1rem;
    }
    .result-card h2, .result-card h3, .result-card p {
        margin-top: 0;
        font-weight: 900;
    }
    .result-normal {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 9px solid #16a34a;
    }
    .result-normal, .result-normal * { color: #14532d !important; }
    .result-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 9px solid #d97706;
    }
    .result-medium, .result-medium * { color: #78350f !important; }
    .result-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 9px solid #dc2626;
    }
    .result-high, .result-high * { color: #7f1d1d !important; }

    .triage-box {
        padding: 1.5rem;
        border-radius: 24px;
        margin-top: 0.6rem;
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.18);
    }
    .triage-box, .triage-box * { color: #ffffff !important; }
    .triage-low { background: linear-gradient(135deg, #16a34a 0%, #15803d 100%); }
    .triage-medium { background: linear-gradient(135deg, #d97706 0%, #b45309 100%); }
    .triage-high { background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); }
    .triage-urgent { background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%); }
    .triage-box h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 950;
    }
    .triage-box p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        font-weight: 700;
    }

    /* Métricas, tablas y expanders */
    div[data-testid="stMetric"],
    [data-testid="stExpander"],
    [data-testid="stDataFrame"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06) !important;
    }
    div[data-testid="stMetric"] * {
        color: var(--text) !important;
    }

    .small-note {
        font-size: 0.84rem;
        color: var(--text-soft) !important;
        line-height: 1.45;
    }

    /* Ocultar chrome extra */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    

    /* =======================================================
       FIX FINAL DE LEGIBILIDAD
       Este bloque va al final para ganar prioridad sobre tema oscuro.
       ======================================================= */
    html, body, .stApp, [data-testid="stAppViewContainer"], main {
        color-scheme: light !important;
    }

    main, main * {
        text-shadow: none !important;
    }

    /* Labels de widgets: sin fondo morado, texto oscuro */
    main [data-testid="stWidgetLabel"],
    main [data-testid="stWidgetLabel"] *,
    main label,
    main label *,
    main label p,
    main label span,
    main [data-testid="stNumberInput"] label *,
    main [data-testid="stSelectbox"] label *,
    main [data-testid="stFileUploader"] label *,
    main [data-testid="stCheckbox"] label *,
    main [data-testid="stRadio"] label * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        background: transparent !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }

    /* Evita que una selección accidental se vea como diseño morado */
    main ::selection {
        background: #dbeafe !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    /* Inputs BaseWeb / Streamlit: fondo blanco, texto oscuro */
    main [data-baseweb="input"],
    main [data-baseweb="input"] > div,
    main [data-baseweb="input"] input,
    main [data-testid="stNumberInput"] div,
    main [data-testid="stNumberInput"] input,
    main input,
    main textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        border-color: #cbd5e1 !important;
        caret-color: #111827 !important;
        opacity: 1 !important;
    }

    /* Botones +/- de number_input */
    main [data-testid="stNumberInput"] button,
    main [data-testid="stNumberInput"] button *,
    main [data-baseweb="input"] button,
    main [data-baseweb="input"] button * {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        border-color: #cbd5e1 !important;
    }

    /* Selectbox */
    main [data-baseweb="select"],
    main [data-baseweb="select"] > div,
    main [data-baseweb="select"] div,
    main [data-baseweb="select"] span,
    main [data-testid="stSelectbox"] div,
    main [data-testid="stSelectbox"] span {
        background-color: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        border-color: #cbd5e1 !important;
        opacity: 1 !important;
    }

    /* Checkboxes: texto oscuro y casilla visible */
    main [data-testid="stCheckbox"],
    main [data-testid="stCheckbox"] *,
    main [data-testid="stCheckbox"] p,
    main [data-testid="stCheckbox"] span,
    main [data-testid="stCheckbox"] label,
    main [data-testid="stCheckbox"] label * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        background: transparent !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    main [data-testid="stCheckbox"] svg,
    main [data-testid="stCheckbox"] input {
        color: #087f8c !important;
        fill: #087f8c !important;
        accent-color: #087f8c !important;
    }

    /* File uploader: no oscuro */
    main [data-testid="stFileUploader"] section,
    main [data-testid="stFileUploader"] section *,
    main [data-testid="stFileUploader"] div,
    main [data-testid="stFileUploader"] span,
    main [data-testid="stFileUploader"] p {
        background-color: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        border-color: #cbd5e1 !important;
        opacity: 1 !important;
    }

    /* Textos secundarios que antes salían casi blancos */
    main .small-note,
    main .card-subtitle,
    main .stCaptionContainer,
    main [data-testid="stCaptionContainer"],
    main [data-testid="stCaptionContainer"] * {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        opacity: 1 !important;
    }

    /* Botón principal con texto blanco real */
    main .stButton > button,
    main .stButton > button *,
    main [data-testid="stDownloadButton"] button,
    main [data-testid="stDownloadButton"] button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background: linear-gradient(90deg, #087f8c, #0284c7) !important;
        font-weight: 900 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------
# UTILIDADES DE CARGA Y PREDICCIÓN
# ------------------------------------------------
def find_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None

@st.cache_resource(show_spinner=False)
def load_model_artifact(path_str):
    artifact = joblib.load(path_str)

    if isinstance(artifact, dict):
        model = artifact.get("model") or artifact.get("pipeline") or artifact.get("best_model")
        feature_names = artifact.get("feature_names") or artifact.get("features")
        class_names = artifact.get("class_names") or artifact.get("labels") or artifact.get("target_names")
        model_name = artifact.get("model_name") or artifact.get("name") or "Modelo entrenado"
        metrics = artifact.get("metrics") or artifact.get("metricas")
    else:
        model = artifact
        feature_names = getattr(model, "feature_names_in_", None)
        class_names = None
        model_name = artifact.__class__.__name__
        metrics = None

    if model is None:
        raise ValueError("El archivo .pkl no contiene un modelo válido. Debe incluir la clave 'model'.")

    if feature_names is not None:
        feature_names = list(feature_names)

    return {
        "model": model,
        "feature_names": feature_names,
        "class_names": class_names,
        "model_name": model_name,
        "metrics": metrics,
        "path": path_str,
    }


def normalize_key(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def class_label(pred, class_names=None, kind="fetal"):
    """Convierte clases numéricas o texto a etiquetas clínicas legibles."""
    raw = pred

    if class_names is not None:
        try:
            if isinstance(class_names, dict):
                if raw in class_names:
                    raw = class_names[raw]
                elif str(raw) in class_names:
                    raw = class_names[str(raw)]
                elif int(raw) in class_names:
                    raw = class_names[int(raw)]
            elif isinstance(class_names, (list, tuple)):
                # Si las clases son 1,2,3, se intenta acceder por índice clase-1.
                try:
                    idx = int(raw) - 1
                    if 0 <= idx < len(class_names):
                        raw = class_names[idx]
                except Exception:
                    pass
        except Exception:
            pass

    s = str(raw).strip()
    ns = normalize_key(s)

    if kind == "fetal":
        if ns in {"1", "normal", "n"}:
            return "Normal"
        if ns in {"2", "suspect", "sospechoso", "s"}:
            return "Sospechoso"
        if ns in {"3", "pathological", "patologico", "patológica", "p"}:
            return "Patológico"
        return s

    if kind == "maternal":
        if "high" in ns or "alto" in ns or ns in {"3", "highrisk"}:
            return "Alto"
        if "mid" in ns or "medium" in ns or "medio" in ns or ns in {"2", "midrisk"}:
            return "Medio"
        if "low" in ns or "bajo" in ns or ns in {"1", "lowrisk"}:
            return "Bajo"
        return s

    return s


def prediction_with_proba(model, X):
    pred = model.predict(X)[0]
    proba_df = None
    confidence = None

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            classes = list(model.classes_) if hasattr(model, "classes_") else list(range(len(proba)))
            proba_df = pd.DataFrame({"Clase": classes, "Probabilidad": proba})
            confidence = float(np.max(proba))
        except Exception:
            pass

    return pred, confidence, proba_df


def probability_table(proba_df, class_names, kind):
    if proba_df is None or proba_df.empty:
        return None
    df = proba_df.copy()
    df["Etiqueta"] = df["Clase"].apply(lambda x: class_label(x, class_names, kind))
    df["Probabilidad"] = (df["Probabilidad"] * 100).round(2)
    return df[["Etiqueta", "Probabilidad"]].sort_values("Probabilidad", ascending=False)


def status_box(title, ok=True, detail=""):
    css = "status-ok" if ok else "status-missing"
    icon = "✅" if ok else "❌"
    st.markdown(
        f"""
        <div class="{css}">
            {icon} {title}<br>
            <span style="font-size:0.85rem;font-weight:500;">{detail}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(title, value, detail="", level="normal"):
    css = {
        "normal": "result-normal",
        "medium": "result-medium",
        "high": "result-high",
    }.get(level, "result-normal")
    st.markdown(
        f"""
        <div class="result-card {css}">
            <h3>{title}</h3>
            <h2>{value}</h2>
            <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def level_from_maternal(label):
    n = normalize_key(label)
    if "alto" in n or "high" in n:
        return "high"
    if "medio" in n or "mid" in n or "medium" in n:
        return "medium"
    return "normal"


def level_from_fetal(label):
    n = normalize_key(label)
    if "patologico" in n or "pathological" in n:
        return "high"
    if "sospechoso" in n or "suspect" in n:
        return "medium"
    return "normal"


def triage_decision(maternal_label=None, fetal_label=None, emergency_flags=None, gestational_age=None, fhr_basal=None, maternal_age=None):
    emergency_flags = emergency_flags or []
    reasons = []
    priority = "Baja"
    css = "triage-low"
    action = "Control y seguimiento según protocolo local."

    maternal_norm = normalize_key(maternal_label or "")
    fetal_norm = normalize_key(fetal_label or "")

    if emergency_flags:
        priority = "Referencia urgente"
        css = "triage-urgent"
        action = "Derivar inmediatamente a un establecimiento de mayor capacidad."
        reasons.extend(emergency_flags)

    if "patologico" in fetal_norm or "pathological" in fetal_norm:
        priority = "Referencia urgente"
        css = "triage-urgent"
        action = "Derivar inmediatamente por resultado fetal patológico."
        reasons.append("Estado fetal clasificado como patológico.")

    elif "alto" in maternal_norm or "high" in maternal_norm:
        if priority != "Referencia urgente":
            priority = "Alta"
            css = "triage-high"
            action = "Priorizar evaluación médica y considerar referencia según disponibilidad."
        reasons.append("Riesgo materno alto.")

    elif "sospechoso" in fetal_norm or "suspect" in fetal_norm:
        if priority not in {"Referencia urgente", "Alta"}:
            priority = "Media"
            css = "triage-medium"
            action = "Repetir evaluación, vigilar evolución y escalar si aparecen signos de alarma."
        reasons.append("Estado fetal sospechoso.")

    elif "medio" in maternal_norm or "mid" in maternal_norm or "medium" in maternal_norm:
        if priority not in {"Referencia urgente", "Alta"}:
            priority = "Media"
            css = "triage-medium"
            action = "Seguimiento cercano y evaluación clínica según protocolo."
        reasons.append("Riesgo materno medio.")

    if fhr_basal is not None:
        try:
            fhr = float(fhr_basal)
            if fhr < 110 or fhr > 160:
                if priority == "Baja":
                    priority = "Media"
                    css = "triage-medium"
                    action = "Repetir medición y evaluar por profesional de salud."
                reasons.append("Frecuencia cardiaca fetal basal fuera del rango 110–160 lpm.")
        except Exception:
            pass

    # Regla clínica de seguridad: gestante menor de edad.
    # No modifica la predicción del modelo, solo eleva la prioridad final de triaje.
    if maternal_age is not None:
        try:
            ma = float(maternal_age)
            if ma < 18:
                if priority in {"Baja", "Media"}:
                    priority = "Alta"
                    css = "triage-high"
                    action = "Priorizar evaluación médica/obstétrica por embarazo en menor de edad."
                reasons.append("Edad materna menor de 18 años: requiere evaluación prioritaria y seguimiento obstétrico.")
        except Exception:
            pass

    if gestational_age is not None:
        try:
            ga = float(gestational_age)
            if ga < 28:
                reasons.append("Edad gestacional menor de 28 semanas: interpretar CTG con cautela.")
            elif ga >= 41:
                if priority == "Baja":
                    priority = "Media"
                    css = "triage-medium"
                reasons.append("Embarazo prolongado: requiere vigilancia clínica.")
        except Exception:
            pass

    if not reasons:
        reasons.append("No se identificaron alertas mayores con los datos ingresados.")

    return priority, css, action, reasons


def show_triage(priority, css, action, reasons):
    icon = "🟢"
    if priority == "Media":
        icon = "🟡"
    elif priority == "Alta":
        icon = "🟠"
    elif priority == "Referencia urgente":
        icon = "🚨"

    st.markdown(
        f"""
        <div class="triage-box {css}">
            <h1>{icon} PRIORIDAD FINAL: {priority.upper()}</h1>
            <p>{action}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Ver motivos de la decisión"):
        for r in reasons:
            st.write(f"• {r}")


def show_cognitive_service(maternal_label, fetal_label, emergency_flags):
    service_output = cognitive_triage_service(
        maternal_prediction=maternal_label,
        fetal_prediction=fetal_label,
        warning_signs=emergency_flags,
    )
    with st.expander("Servicio cognitivo interno: explicación de triaje"):
        st.write(f"Prioridad sugerida por reglas transparentes: **{service_output['priority']}**")
        for explanation in service_output["explanations"]:
            st.write(f"• {explanation}")
        st.caption(service_output["ethical_warning"])
    return service_output

# ------------------------------------------------
# MAPEO DE VARIABLES CTG
# ------------------------------------------------
FETAL_ALIASES = {
    "LB": ["LB", "baseline_value", "baseline value", "baseline", "fhr_baseline"],
    "AC": ["AC", "accelerations", "acceleration"],
    "FM": ["FM", "fetal_movement", "fetal movement"],
    "UC": ["UC", "uterine_contractions", "uterine contractions"],
    "DL": ["DL", "light_decelerations", "light decelerations"],
    "DS": ["DS", "severe_decelerations", "severe decelerations"],
    "DP": ["DP", "prolongued_decelerations", "prolonged_decelerations", "prolongued decelerations", "prolonged decelerations"],
    "ASTV": ["ASTV", "abnormal_short_term_variability", "abnormal short term variability"],
    "MSTV": ["MSTV", "mean_short_term_variability", "mean short term variability"],
    "ALTV": ["ALTV", "pct_abnormal_long_term_variability", "percentage_of_time_with_abnormal_long_term_variability", "abnormal_long_term_variability"],
    "MLTV": ["MLTV", "mean_long_term_variability", "mean long term variability"],
    "Width": ["Width", "hist_width", "histogram_width", "histogram width"],
    "Min": ["Min", "hist_min", "histogram_min", "histogram min"],
    "Max": ["Max", "hist_max", "histogram_max", "histogram max"],
    "Nmax": ["Nmax", "hist_n_peaks", "histogram_number_of_peaks", "number_of_histogram_peaks"],
    "Nzeros": ["Nzeros", "hist_n_zeros", "histogram_number_of_zeroes", "histogram_number_of_zeros"],
    "Mode": ["Mode", "hist_mode", "histogram_mode", "histogram mode"],
    "Mean": ["Mean", "hist_mean", "histogram_mean", "histogram mean"],
    "Median": ["Median", "hist_median", "histogram_median", "histogram median"],
    "Variance": ["Variance", "hist_variance", "histogram_variance", "histogram variance"],
    "Tendency": ["Tendency", "hist_tendency", "histogram_tendency", "histogram tendency"],
}

# Alias inverso para nombres tipo Kaggle si el modelo fue entrenado con columnas largas.
KAGGLE_TO_UCI = {
    "baseline_value": "LB",
    "accelerations": "AC",
    "fetal_movement": "FM",
    "uterine_contractions": "UC",
    "light_decelerations": "DL",
    "severe_decelerations": "DS",
    "prolongued_decelerations": "DP",
    "prolonged_decelerations": "DP",
    "abnormal_short_term_variability": "ASTV",
    "mean_short_term_variability": "MSTV",
    "pct_abnormal_long_term_variability": "ALTV",
    "percentage_of_time_with_abnormal_long_term_variability": "ALTV",
    "mean_long_term_variability": "MLTV",
    "hist_width": "Width",
    "histogram_width": "Width",
    "hist_min": "Min",
    "histogram_min": "Min",
    "hist_max": "Max",
    "histogram_max": "Max",
    "hist_n_peaks": "Nmax",
    "histogram_number_of_peaks": "Nmax",
    "hist_n_zeros": "Nzeros",
    "histogram_number_of_zeroes": "Nzeros",
    "histogram_number_of_zeros": "Nzeros",
    "hist_mode": "Mode",
    "histogram_mode": "Mode",
    "hist_mean": "Mean",
    "histogram_mean": "Mean",
    "hist_median": "Median",
    "histogram_median": "Median",
    "hist_variance": "Variance",
    "histogram_variance": "Variance",
    "hist_tendency": "Tendency",
    "histogram_tendency": "Tendency",
}


def aliases_for_feature(feature):
    if feature in FETAL_ALIASES:
        return FETAL_ALIASES[feature]
    # Si el feature del modelo está en formato Kaggle, añadir aliases del UCI equivalente.
    norm_f = normalize_key(feature)
    for kaggle, uci in KAGGLE_TO_UCI.items():
        if normalize_key(kaggle) == norm_f and uci in FETAL_ALIASES:
            return [feature] + FETAL_ALIASES[uci]
    # Si no hay alias, usar el nombre exacto.
    return [feature]


def prepare_fetal_dataframe(df_raw, feature_names):
    feature_names = feature_names or DEFAULT_FETAL_FEATURES_UCI
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    normalized_columns = {normalize_key(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    missing = []

    for feature in feature_names:
        found_col = None
        for alias in aliases_for_feature(feature):
            key = normalize_key(alias)
            if key in normalized_columns:
                found_col = normalized_columns[key]
                break
        if found_col is None:
            missing.append(feature)
        else:
            out[feature] = pd.to_numeric(df[found_col], errors="coerce")

    if missing:
        raise ValueError("Faltan columnas para el modelo fetal: " + ", ".join(missing))

    if out.isnull().any().any():
        bad_cols = list(out.columns[out.isnull().any()])
        raise ValueError("Hay valores vacíos o no numéricos en: " + ", ".join(bad_cols))

    return out[feature_names]


def fetal_manual_defaults(feature):
    defaults = {
        "LB": 140, "AC": 0.003, "FM": 0.0, "UC": 0.002, "DL": 0.0, "DS": 0.0, "DP": 0.0,
        "ASTV": 25, "MSTV": 1.5, "ALTV": 5, "MLTV": 8, "Width": 60, "Min": 120, "Max": 160,
        "Nmax": 2, "Nzeros": 0, "Mode": 140, "Mean": 140, "Median": 140, "Variance": 5, "Tendency": 0,
    }
    if feature in defaults:
        return defaults[feature]

    # Para nombres Kaggle, buscar equivalente UCI.
    for kaggle, uci in KAGGLE_TO_UCI.items():
        if normalize_key(feature) == normalize_key(kaggle):
            return defaults.get(uci, 0.0)
    return 0.0

# ------------------------------------------------
# CARGA DE MODELOS
# ------------------------------------------------
maternal_path = find_first_existing(MATERNAL_MODEL_PATHS)
fetal_path = find_first_existing(FETAL_MODEL_PATHS)
fetal_mlp_path = find_first_existing(FETAL_MLP_MODEL_PATHS)

maternal_artifact = None
fetal_artifact = None
fetal_mlp_artifact = None
maternal_error = None
fetal_error = None
fetal_mlp_error = None

if maternal_path:
    try:
        maternal_artifact = load_model_artifact(str(maternal_path))
    except Exception as e:
        maternal_error = str(e)

if fetal_path:
    try:
        fetal_artifact = load_model_artifact(str(fetal_path))
    except Exception as e:
        fetal_error = str(e)

if fetal_mlp_path:
    try:
        fetal_mlp_artifact = load_model_artifact(str(fetal_mlp_path))
    except Exception as e:
        fetal_mlp_error = str(e)

maternal_features = (
    maternal_artifact.get("feature_names") if maternal_artifact else None
) or DEFAULT_MATERNAL_FEATURES

# ------------------------------------------------
# ENCABEZADO Y SIDEBAR
# ------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🏥 MaternIA Integral</h1>
        <p>Triaje materno-fetal preventivo para postas médicas rurales</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="safe-banner">
        ⚠️ Prototipo académico. No emite diagnóstico médico ni reemplaza la evaluación de un profesional de salud.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🧭 Navegación")
    mode = st.radio(
        "Selecciona un módulo",
        ["Evaluación integral", "Modo posta", "CTG procesado", "Avanzado académico", "Reportes"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Modelos")
    st.write("🩺 Riesgo materno")
    st.write("👶 Estado fetal CTG")
    fetal_model_option = st.selectbox(
        "Modelo fetal a utilizar",
        ["Modelo optimizado", "Red neuronal MLP"],
        disabled=not fetal_mlp_artifact,
        help="La red neuronal MLP aparece cuando existe models/maternIA_fetal_mlp_neural_network.pkl.",
    )
    st.markdown("---")
    st.caption("MaternIA vFinal | Uso académico")

if fetal_model_option == "Red neuronal MLP" and fetal_mlp_artifact:
    fetal_artifact = fetal_mlp_artifact
    fetal_path = fetal_mlp_path

fetal_features = (
    fetal_artifact.get("feature_names") if fetal_artifact else None
) or DEFAULT_FETAL_FEATURES_UCI

# ------------------------------------------------
# PANEL DE ESTADO
# ------------------------------------------------
col_s1, col_s2 = st.columns(2)
with col_s1:
    status_box(
        "Modelo materno cargado" if maternal_artifact else "Modelo materno no encontrado",
        ok=bool(maternal_artifact),
        detail=str(maternal_path.name) if maternal_artifact else (maternal_error or "Coloca maternIA_maternal_risk_model.pkl en models/"),
    )
with col_s2:
    status_box(
        "Modelo fetal cargado" if fetal_artifact else "Modelo fetal no encontrado",
        ok=bool(fetal_artifact),
        detail=str(fetal_path.name) if fetal_artifact else (fetal_error or "Coloca maternIA_fetal_health_model.pkl en models/"),
    )

st.write("")

# ------------------------------------------------
# FORMULARIO MATERNO REUTILIZABLE
# ------------------------------------------------
def maternal_form(prefix="maternal"):
    st.markdown('<div class="card"><div class="card-title">🩺 Evaluación materna</div><div class="card-subtitle">Variables usadas por el modelo materno y datos clínicos de apoyo.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Edad materna", 10, 55, 28, key=f"{prefix}_age")
        gestational_age = st.number_input("Edad gestacional (semanas)", 4, 42, 32, key=f"{prefix}_ga")
        heart_rate = st.number_input("FC materna (lpm)", 40, 180, 82, key=f"{prefix}_hr")
    with c2:
        systolic = st.number_input("Presión sistólica (mmHg)", 70, 220, 120, key=f"{prefix}_sbp")
        diastolic = st.number_input("Presión diastólica (mmHg)", 40, 140, 80, key=f"{prefix}_dbp")
        temp_c = st.number_input("Temperatura materna (°C)", 34.0, 42.0, 36.7, step=0.1, key=f"{prefix}_temp")
    with c3:
        glucose_unit = st.selectbox("Unidad de glucosa", ["mmol/L", "mg/dL"], key=f"{prefix}_glucose_unit")
        glucose_value = st.number_input("Glucosa", 1.0, 400.0, 7.0 if glucose_unit == "mmol/L" else 126.0, step=0.1, key=f"{prefix}_glucose")
        fhr_basal = st.number_input("FC fetal basal si está disponible (lpm)", 50, 220, 140, key=f"{prefix}_fhr")

    st.markdown("#### Signos de alarma")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        bleeding = st.checkbox("Sangrado vaginal", key=f"{prefix}_bleeding")
    with a2:
        liquid_loss = st.checkbox("Pérdida de líquido", key=f"{prefix}_liquid")
    with a3:
        seizures = st.checkbox("Convulsiones", key=f"{prefix}_seizures")
    with a4:
        severe_pain = st.checkbox("Dolor intenso", key=f"{prefix}_pain")

    decreased_movements = st.checkbox("Disminución marcada de movimientos fetales", key=f"{prefix}_movements")

    st.markdown('</div>', unsafe_allow_html=True)

    body_temp_f = temp_c * 9 / 5 + 32
    bs_mmol = glucose_value if glucose_unit == "mmol/L" else glucose_value / 18.0

    emergency_flags = []
    if bleeding:
        emergency_flags.append("Sangrado vaginal referido.")
    if liquid_loss:
        emergency_flags.append("Pérdida de líquido referida.")
    if seizures:
        emergency_flags.append("Convulsiones referidas.")
    if severe_pain:
        emergency_flags.append("Dolor intenso referido.")
    if decreased_movements:
        emergency_flags.append("Disminución marcada de movimientos fetales referida.")

    maternal_input = pd.DataFrame([{
        "Age": age,
        "SystolicBP": systolic,
        "DiastolicBP": diastolic,
        "BS": bs_mmol,
        "BodyTemp": body_temp_f,
        "HeartRate": heart_rate,
    }])

    # Reordenar según features reales del modelo.
    for f in maternal_features:
        if f not in maternal_input.columns:
            maternal_input[f] = 0
    maternal_input = maternal_input[maternal_features]

    context = {
        "age": age,
        "gestational_age": gestational_age,
        "heart_rate": heart_rate,
        "systolic": systolic,
        "diastolic": diastolic,
        "temp_c": temp_c,
        "glucose_mmol": bs_mmol,
        "fhr_basal": fhr_basal,
        "emergency_flags": emergency_flags,
    }
    return maternal_input, context


def run_maternal_prediction(maternal_input):
    if not maternal_artifact:
        return None, None, None
    model = maternal_artifact["model"]
    pred, conf, proba = prediction_with_proba(model, maternal_input)
    label = class_label(pred, maternal_artifact.get("class_names"), kind="maternal")
    return label, conf, proba


def run_fetal_prediction(fetal_input):
    if not fetal_artifact:
        return None, None, None
    model = fetal_artifact["model"]
    pred, conf, proba = prediction_with_proba(model, fetal_input)
    label = class_label(pred, fetal_artifact.get("class_names"), kind="fetal")
    return label, conf, proba

# ------------------------------------------------
# MODO 1: EVALUACIÓN INTEGRAL
# ------------------------------------------------
if mode == "Evaluación integral":
    st.subheader("🧾 Evaluación integral materno-fetal")
    st.caption("Combina el modelo materno, el modelo fetal CTG cuando hay archivo procesado y signos de alarma.")

    maternal_input, ctx = maternal_form(prefix="integral")

    st.markdown('<div class="card"><div class="card-title">👶 Evaluación fetal CTG opcional</div><div class="card-subtitle">Carga un CSV procesado con variables CTG. Si no tienes CTG, la decisión se hará con riesgo materno y signos de alarma.</div>', unsafe_allow_html=True)
    uploaded_ctg = st.file_uploader("Sube archivo CTG procesado (.csv)", type=["csv"], key="integral_ctg")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚑 Calcular triaje integral", use_container_width=True):
        maternal_label, maternal_conf, maternal_proba = run_maternal_prediction(maternal_input)

        fetal_label = None
        fetal_conf = None
        fetal_proba = None
        fetal_input_preview = None

        if uploaded_ctg is not None and fetal_artifact:
            try:
                df_ctg = pd.read_csv(uploaded_ctg)
                fetal_input = prepare_fetal_dataframe(df_ctg, fetal_features)
                fetal_input_preview = fetal_input.head(1)
                fetal_label, fetal_conf, fetal_proba = run_fetal_prediction(fetal_input.head(1))
            except Exception as e:
                st.error(f"No se pudo procesar el CTG: {e}")

        r1, r2, r3 = st.columns(3)
        with r1:
            if maternal_label:
                result_card(
                    "Riesgo materno",
                    maternal_label,
                    f"Confianza: {maternal_conf*100:.1f}%" if maternal_conf is not None else "Modelo materno aplicado",
                    level_from_maternal(maternal_label),
                )
            else:
                result_card("Riesgo materno", "No disponible", "Modelo materno no cargado", "medium")

        with r2:
            if fetal_label:
                result_card(
                    "Estado fetal CTG",
                    fetal_label,
                    f"Confianza: {fetal_conf*100:.1f}%" if fetal_conf is not None else "Modelo fetal aplicado",
                    level_from_fetal(fetal_label),
                )
            else:
                result_card("Estado fetal CTG", "No aplicado", "No se cargó CTG procesado", "medium")

        with r3:
            st.metric("Edad gestacional", f"{ctx['gestational_age']} sem")
            st.metric("FC fetal basal", f"{ctx['fhr_basal']} lpm")

        priority, css, action, reasons = triage_decision(
            maternal_label=maternal_label,
            fetal_label=fetal_label,
            emergency_flags=ctx["emergency_flags"],
            gestational_age=ctx["gestational_age"],
            fhr_basal=ctx["fhr_basal"],
            maternal_age=ctx["age"],
        )
        show_triage(priority, css, action, reasons)
        cognitive_output = show_cognitive_service(maternal_label, fetal_label, ctx["emergency_flags"])

        with st.expander("📊 Ver probabilidades del modelo materno"):
            table = probability_table(maternal_proba, maternal_artifact.get("class_names") if maternal_artifact else None, "maternal")
            if table is not None:
                st.dataframe(table, use_container_width=True)
            else:
                st.info("El modelo materno no entregó probabilidades.")

        if fetal_label:
            with st.expander("📊 Ver probabilidades del modelo fetal"):
                table = probability_table(fetal_proba, fetal_artifact.get("class_names") if fetal_artifact else None, "fetal")
                if table is not None:
                    st.dataframe(table, use_container_width=True)
                else:
                    st.info("El modelo fetal no entregó probabilidades.")

        report = {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "riesgo_materno": maternal_label,
            "estado_fetal": fetal_label or "No aplicado",
            "prioridad_final": priority,
            "accion_recomendada": action,
            "motivos": reasons,
            "servicio_cognitivo": cognitive_output,
            "nota": "Prototipo académico; no reemplaza evaluación profesional."
        }
        st.download_button(
            "⬇️ Descargar reporte de triaje (.json)",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="reporte_maternia_integral.json",
            mime="application/json",
            use_container_width=True,
        )

# ------------------------------------------------
# MODO 2: POSTA / MATERNO
# ------------------------------------------------
elif mode == "Modo posta":
    st.subheader("🩺 Modo posta: evaluación materna rápida")
    st.caption("Usa edad materna, presión, glucosa, temperatura, FC materna, edad gestacional y signos de alarma.")

    maternal_input, ctx = maternal_form(prefix="posta")

    if st.button("Evaluar riesgo materno", use_container_width=True):
        maternal_label, maternal_conf, maternal_proba = run_maternal_prediction(maternal_input)
        c1, c2 = st.columns([1, 1])
        with c1:
            if maternal_label:
                result_card(
                    "Riesgo materno",
                    maternal_label,
                    f"Confianza: {maternal_conf*100:.1f}%" if maternal_conf is not None else "Modelo aplicado",
                    level_from_maternal(maternal_label),
                )
            else:
                st.error("No se encontró el modelo materno.")

        with c2:
            priority, css, action, reasons = triage_decision(
                maternal_label=maternal_label,
                fetal_label=None,
                emergency_flags=ctx["emergency_flags"],
                gestational_age=ctx["gestational_age"],
                fhr_basal=ctx["fhr_basal"],
                maternal_age=ctx["age"],
            )
            show_triage(priority, css, action, reasons)
            show_cognitive_service(maternal_label, None, ctx["emergency_flags"])

        with st.expander("Ver probabilidades"):
            table = probability_table(maternal_proba, maternal_artifact.get("class_names") if maternal_artifact else None, "maternal")
            if table is not None:
                st.dataframe(table, use_container_width=True)

# ------------------------------------------------
# MODO 3: CTG PROCESADO
# ------------------------------------------------
elif mode == "CTG procesado":
    st.subheader("👶 Clasificación fetal con CTG procesado")
    st.caption("Carga un CSV con las 21 variables cardiotocográficas. Se aceptan nombres UCI o Kaggle cuando son equivalentes.")

    example_path = DATA_DIR / "ejemplo_ctg_procesado.csv"
    if example_path.exists():
        with open(example_path, "rb") as f:
            st.download_button(
                "⬇️ Descargar CSV de ejemplo",
                data=f,
                file_name="ejemplo_ctg_procesado.csv",
                mime="text/csv",
            )

    uploaded = st.file_uploader("Sube archivo CTG procesado (.csv)", type=["csv"], key="ctg_upload")

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            st.markdown("### Vista previa del archivo")
            st.dataframe(df_raw.head(), use_container_width=True)

            fetal_input = prepare_fetal_dataframe(df_raw, fetal_features)

            if fetal_artifact:
                model = fetal_artifact["model"]
                preds = model.predict(fetal_input)
                result_labels = [class_label(p, fetal_artifact.get("class_names"), "fetal") for p in preds]
                result_df = df_raw.copy()
                result_df["estado_fetal_predicho"] = result_labels

                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(fetal_input)
                    result_df["confianza_max_%"] = (np.max(probas, axis=1) * 100).round(2)

                st.markdown("### Resultados")
                st.dataframe(result_df, use_container_width=True)

                first_label = result_labels[0]
                first_conf = result_df.loc[result_df.index[0], "confianza_max_%"] if "confianza_max_%" in result_df.columns else None
                result_card(
                    "Resultado principal de la primera fila",
                    first_label,
                    f"Confianza: {first_conf:.1f}%" if first_conf is not None else "Modelo fetal aplicado",
                    level_from_fetal(first_label),
                )

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Descargar resultados (.csv)",
                    data=csv_out,
                    file_name="resultados_ctg_maternia.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.error("No se encontró el modelo fetal.")

        except Exception as e:
            st.error(f"No se pudo procesar el archivo: {e}")
            st.info("Verifica que el CSV tenga las 21 variables CTG procesadas.")

# ------------------------------------------------
# MODO 4: AVANZADO ACADÉMICO
# ------------------------------------------------
elif mode == "Avanzado académico":
    st.subheader("🔬 Modo avanzado académico")
    st.caption("Permite ingresar manualmente las variables CTG para demostrar cómo responde el modelo. No es el modo rutinario para una posta.")

    st.markdown('<div class="card"><div class="card-title">Variables CTG</div><div class="card-subtitle">Valores de ejemplo basados en rangos habituales del dataset procesado.</div>', unsafe_allow_html=True)
    manual_values = {}
    cols = st.columns(3)
    for i, feature in enumerate(fetal_features):
        with cols[i % 3]:
            default = float(fetal_manual_defaults(feature))
            manual_values[feature] = st.number_input(str(feature), value=default, step=0.001 if default < 10 else 1.0, key=f"adv_{feature}")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Clasificar CTG manual", use_container_width=True):
        fetal_input = pd.DataFrame([manual_values])[fetal_features]
        fetal_label, fetal_conf, fetal_proba = run_fetal_prediction(fetal_input)
        if fetal_label:
            result_card(
                "Estado fetal CTG",
                fetal_label,
                f"Confianza: {fetal_conf*100:.1f}%" if fetal_conf is not None else "Modelo fetal aplicado",
                level_from_fetal(fetal_label),
            )
            table = probability_table(fetal_proba, fetal_artifact.get("class_names") if fetal_artifact else None, "fetal")
            if table is not None:
                st.dataframe(table, use_container_width=True)
        else:
            st.error("No se encontró el modelo fetal.")

# ------------------------------------------------
# MODO 5: REPORTES
# ------------------------------------------------
elif mode == "Reportes":
    st.subheader("📊 Reportes del entrenamiento")
    st.caption("Muestra las evidencias generadas por el entrenamiento: métricas, matriz de confusión e importancia de variables.")

    metrics_candidates = [
        REPORTS_DIR / "metrics_report_ordered.csv",
        REPORTS_DIR / "metrics_report.csv",
        REPORTS_DIR / "maternal_metrics_report_ordered.csv",
        REPORTS_DIR / "maternal_metrics_report.csv",
        REPORTS_DIR / "fetal_metrics_report_ordered.csv",
        REPORTS_DIR / "fetal_metrics_report.csv",
        REPORTS_DIR / "fetal_mlp_metrics.json",
    ]

    shown_any = False
    for mpath in metrics_candidates:
        if mpath.exists():
            st.markdown(f"### {mpath.name}")
            try:
                if mpath.suffix == ".json":
                    st.json(json.loads(mpath.read_text(encoding="utf-8")))
                else:
                    st.dataframe(pd.read_csv(mpath), use_container_width=True)
                shown_any = True
            except Exception:
                st.warning(f"No se pudo abrir {mpath.name}")

    image_candidates = [
        REPORTS_DIR / "confusion_matrix.png",
        REPORTS_DIR / "feature_importance.png",
        REPORTS_DIR / "shap_summary.png",
        REPORTS_DIR / "maternal_confusion_matrix.png",
        REPORTS_DIR / "fetal_confusion_matrix.png",
        REPORTS_DIR / "fetal_mlp_confusion_matrix.png",
    ]

    for ipath in image_candidates:
        if ipath.exists():
            st.markdown(f"### {ipath.name}")
            st.image(str(ipath), use_container_width=True)
            shown_any = True

    if not shown_any:
        st.info("Todavía no hay reportes en la carpeta reports/. Ejecuta el entrenamiento o copia los resultados desde Colab.")

# ------------------------------------------------
# PIE DE PÁGINA
# ------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <p class="small-note">
    MaternIA Integral es un prototipo académico de apoyo al triaje. Sus resultados deben interpretarse como orientación computacional, no como diagnóstico definitivo.
    </p>
    """,
    unsafe_allow_html=True,
)
