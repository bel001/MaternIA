# ============================================================
# app_streamlit_maternia.py
# MATERNIA - APP VISUAL FINAL
#
# Usa 2 modelos:
# 1. maternIA_quick_triage_model.pkl
# 2. maternIA_fetal_health_model.pkl
#
# Mejoras:
# - Solo 3 pestañas: Evaluar paciente, CTG procesado, Información.
# - Probabilidades del modelo ML en porcentaje con 2 decimales.
# - Separación entre probabilidad ML y prioridad final de atención.
# - No se falsifica la probabilidad del modelo.
# - Si hay alerta clínica crítica, la prioridad final puede subir a 100%.
# - Cambia "modelo rápido" por "modelo ML de riesgo materno".
# ============================================================

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURACION
# ============================================================

st.set_page_config(
    page_title="MaternIA - Triaje con ML",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent

QUICK_MODEL_PATH = BASE_DIR / "maternIA_quick_triage_model.pkl"
FETAL_MODEL_PATH = BASE_DIR / "maternIA_fetal_health_model.pkl"


# ============================================================
# CONSTANTES
# ============================================================

QUICK_FEATURE_COLUMNS = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate"
]

QUICK_CLASS_NAMES = {
    0: "Riesgo bajo",
    1: "Riesgo medio",
    2: "Riesgo alto"
}

FETAL_FEATURE_COLUMNS = [
    "baseline value",
    "accelerations",
    "fetal_movement",
    "uterine_contractions",
    "light_decelerations",
    "severe_decelerations",
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
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
    "histogram_tendency"
]

FETAL_CLASS_NAMES = {
    1: "Normal",
    2: "Sospechoso",
    3: "Patológico"
}


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
    .main-header {
        padding: 1.5rem 1.7rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #172554 0%, #0f766e 55%, #14532d 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,.18);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.25rem;
        font-weight: 850;
    }

    .main-header p {
        margin: .5rem 0 0 0;
        color: rgba(255,255,255,.9);
        font-size: 1rem;
    }

    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,.24);
        background: rgba(30,41,59,.38);
        margin-bottom: .75rem;
    }

    .metric-label {
        font-size: .82rem;
        color: #94a3b8;
        margin-bottom: .2rem;
    }

    .metric-value {
        font-size: 1.22rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .info-box {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: rgba(15, 23, 42, .28);
        border: 1px solid rgba(148,163,184,.25);
        margin-bottom: 1rem;
    }

    .important-box {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: rgba(217, 119, 6, .16);
        border-left: 8px solid #f59e0b;
        margin-bottom: 1rem;
    }

    .risk-card {
        padding: 1.25rem 1.4rem;
        border-radius: 22px;
        margin-top: 1rem;
        border: 1px solid rgba(148,163,184,.25);
    }

    .risk-low {
        background: rgba(22, 101, 52, .18);
        border-left: 8px solid #22c55e;
    }

    .risk-medium {
        background: rgba(217, 119, 6, .18);
        border-left: 8px solid #f59e0b;
    }

    .risk-high {
        background: rgba(185, 28, 28, .20);
        border-left: 8px solid #ef4444;
    }

    .risk-none {
        background: rgba(71, 85, 105, .25);
        border-left: 8px solid #94a3b8;
    }

    .priority-box {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: rgba(15, 23, 42, .34);
        border: 1px solid rgba(148,163,184,.25);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .priority-big {
        font-size: 2rem;
        font-weight: 850;
        margin-bottom: .25rem;
    }

    .priority-label {
        color: #cbd5e1;
        font-size: .95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CARGA DE MODELOS
# ============================================================

@st.cache_resource
def load_quick_model():
    if not QUICK_MODEL_PATH.exists():
        st.error(
            "No se encontró maternIA_quick_triage_model.pkl. "
            "Ejecuta el Colab 1 y coloca el archivo en la misma carpeta de la app."
        )
        st.stop()

    artifact = joblib.load(QUICK_MODEL_PATH)

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact

    return {
        "model": artifact,
        "model_name": "Modelo ML de riesgo materno",
        "feature_columns": QUICK_FEATURE_COLUMNS,
        "class_names": QUICK_CLASS_NAMES
    }


@st.cache_resource
def load_fetal_model():
    if not FETAL_MODEL_PATH.exists():
        st.error(
            "No se encontró maternIA_fetal_health_model.pkl. "
            "Ejecuta el Colab 2 y coloca el archivo en la misma carpeta de la app."
        )
        st.stop()

    artifact = joblib.load(FETAL_MODEL_PATH)

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact

    return {
        "model": artifact,
        "model_name": "Modelo fetal CTG",
        "feature_columns": FETAL_FEATURE_COLUMNS,
        "class_names": FETAL_CLASS_NAMES
    }


quick_artifact = load_quick_model()
fetal_artifact = load_fetal_model()

quick_model = quick_artifact["model"]
fetal_model = fetal_artifact["model"]


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def normalize_int(value):
    try:
        return int(value)
    except Exception:
        return int(float(value))


def celsius_to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32


def mgdl_to_mmol_l(glucose_mgdl):
    return glucose_mgdl / 18.018


def get_model_classes(model, default_classes):
    if hasattr(model, "classes_"):
        return model.classes_

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"].classes_

    return np.array(default_classes)


def show_probabilities(probabilities, classes, names):
    probabilities_percent = np.round(probabilities * 100, 2)

    prob_df_numeric = pd.DataFrame({
        "Clase": [names.get(normalize_int(c), str(c)) for c in classes],
        "Probabilidad (%)": probabilities_percent
    })

    prob_df_show = prob_df_numeric.copy()
    prob_df_show["Probabilidad"] = prob_df_show["Probabilidad (%)"].apply(
        lambda x: f"{x:.2f}%"
    )
    prob_df_show = prob_df_show[["Clase", "Probabilidad"]]

    st.subheader("Probabilidades del modelo ML")
    st.caption(
        "Estas probabilidades pertenecen al modelo. "
        "La prioridad final puede subir si existe una alerta clínica de seguridad."
    )

    st.dataframe(
        prob_df_show,
        use_container_width=True,
        hide_index=True
    )

    chart_df = prob_df_numeric.set_index("Clase")
    st.bar_chart(chart_df)


def get_final_priority_percent(final_level, has_clinical_alerts, model_confidence_percent):
    """
    Prioridad final de triaje.
    No es probabilidad del modelo.

    - Alerta clínica crítica: 100%
    - Riesgo alto por ML: mínimo 90%
    - Riesgo medio: mínimo 65%
    - Riesgo bajo: confianza del modelo con tope menor a 90%
    """
    if final_level == "Riesgo alto" and has_clinical_alerts:
        return 100.0

    if final_level == "Riesgo alto":
        return max(90.0, model_confidence_percent)

    if final_level == "Riesgo medio":
        return max(65.0, model_confidence_percent)

    if final_level == "Riesgo bajo":
        return max(0.0, min(model_confidence_percent, 89.99))

    return 0.0


def show_final_priority(final_level, priority_percent, source_text):
    if final_level == "Riesgo alto":
        color = "#ef4444"
    elif final_level == "Riesgo medio":
        color = "#f59e0b"
    elif final_level == "Riesgo bajo":
        color = "#22c55e"
    else:
        color = "#94a3b8"

    st.markdown(
        f"""
        <div class="priority-box">
            <div class="priority-big" style="color:{color};">{priority_percent:.0f}%</div>
            <div class="priority-label"><b>Prioridad final de atención:</b> {final_level}</div>
            <div class="priority-label">{source_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_action_card(title, action, motives, level):
    css = {
        "Riesgo bajo": "risk-low",
        "Riesgo medio": "risk-medium",
        "Riesgo alto": "risk-high",
        "No interpretable": "risk-none",
        "Normal": "risk-low",
        "Sospechoso": "risk-medium",
        "Patológico": "risk-high",
    }.get(level, "risk-none")

    st.markdown(
        f"""
        <div class="risk-card {css}">
            <h2>{title}</h2>
            <p><b>Acción:</b> {action}</p>
            <p><b>Nota:</b> No es diagnóstico definitivo. La decisión final debe ser clínica.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Motivo del resultado")

    for motive in motives:
        if level in ["Riesgo alto", "Patológico"]:
            st.error(motive)
        elif level in ["Riesgo medio", "Sospechoso"]:
            st.warning(motive)
        elif level == "No interpretable":
            st.info(motive)
        else:
            st.success(motive)


def quick_safety_gate(data):
    blocking = []
    warnings = []

    if not data["edad_gestacional_registrada"]:
        blocking.append("No se registró edad gestacional.")

    if not data["motivo_evaluacion"].strip():
        blocking.append("No se registró motivo de evaluación.")

    if not data["latido_confirmado"]:
        blocking.append("No se confirmó que el latido sea fetal y no materno.")

    if data["calidad_senal"] == "Mala / no confiable":
        blocking.append("La señal o medición es mala/no confiable.")

    if data["calidad_senal"] == "Dudosa":
        warnings.append("Señal dudosa: repetir medición si es posible.")

    return blocking, warnings


def clinical_alerts_from_context(data):
    alerts = []
    cautions = []

    if data["sangrado"]:
        alerts.append("Alerta clínica: sangrado vaginal.")

    if data["perdida_liquido"]:
        alerts.append("Alerta clínica: pérdida de líquido.")

    if data["movimientos_disminuidos"]:
        alerts.append("Alerta clínica: disminución de movimientos fetales.")

    if data["temperatura_c"] >= 38.0:
        alerts.append("Alerta clínica: fiebre materna.")

    if data["sistolica"] >= 140 or data["diastolica"] >= 90:
        alerts.append("Alerta clínica: PA ≥140/90.")

    if data["convulsiones"]:
        alerts.append("Alerta clínica: convulsiones o episodio neurológico severo.")

    if data["dolor_intenso"]:
        alerts.append("Alerta clínica: dolor abdominal intenso.")

    if data["cefalea_vision"]:
        alerts.append("Alerta clínica: cefalea intensa, visión borrosa o edema marcado.")

    if data["fhr_disponible"]:
        fhr = data["fhr_baseline"]

        if fhr < 100:
            alerts.append("Alerta clínica: FHR menor a 100 lpm.")

        elif fhr > 160:
            alerts.append("Alerta clínica: FHR mayor a 160 lpm.")

        elif 100 <= fhr < 110:
            cautions.append("Precaución: FHR entre 100 y 109 lpm. Repetir medición.")

    else:
        cautions.append("Precaución: no se registró FHR basal.")

    if data["desaceleraciones"] == "Sí, prolongadas o repetidas":
        alerts.append("Alerta clínica: desaceleraciones prolongadas o repetidas.")

    elif data["desaceleraciones"] == "Sí, leves o aisladas":
        cautions.append("Precaución: desaceleraciones leves o aisladas.")

    elif data["desaceleraciones"] == "No se pudo evaluar":
        cautions.append("Precaución: no se pudo evaluar desaceleraciones.")

    if data["contracciones"] == "Frecuentes / muy dolorosas":
        cautions.append("Precaución: contracciones frecuentes o muy dolorosas.")

    return alerts, cautions


def quick_action_from_prediction(pred_label, alerts, cautions):
    motives = [f"Predicción del modelo ML de riesgo materno: {pred_label}."]

    has_clinical_alerts = len(alerts) > 0

    if has_clinical_alerts:
        level = "Riesgo alto"
        title = "🔴 Resultado: Riesgo alto"
        action = "Derivar o contactar obstetra / centro de mayor capacidad."
        motives.append(
            "La prioridad final sube a Riesgo alto por alerta clínica de seguridad, "
            "aunque el modelo ML haya estimado otro nivel."
        )
        motives.extend(alerts)
        motives.extend(cautions)
        return title, action, motives, level, has_clinical_alerts

    if pred_label == "Riesgo alto":
        title = "🔴 Resultado: Riesgo alto"
        action = "Derivar o contactar obstetra / centro de mayor capacidad."
        motives.extend(cautions)
        return title, action, motives, "Riesgo alto", False

    if pred_label == "Riesgo medio" or cautions:
        title = "🟠 Resultado: Riesgo medio"
        action = "Repetir medición, vigilar evolución y consultar si persiste."
        motives.extend(cautions)
        return title, action, motives, "Riesgo medio", False

    title = "🟢 Resultado: Riesgo bajo"
    action = "Continuar control y reevaluar si aparecen síntomas."
    motives.append("Sin señales clínicas de alarma registradas.")
    return title, action, motives, "Riesgo bajo", False


def build_quick_summary_table(data, ml_input, pred_label, final_level, priority_percent):
    rows = [
        [
            "Calidad de muestra",
            "Edad gestacional registrada",
            "Sí" if data["edad_gestacional_registrada"] else "No",
            "Requisito mínimo."
        ],
        [
            "Calidad de muestra",
            "Motivo de evaluación",
            data["motivo_evaluacion"] or "No registrado",
            "Ayuda a contextualizar."
        ],
        [
            "Calidad de muestra",
            "Latido fetal confirmado",
            "Sí" if data["latido_confirmado"] else "No",
            "Si no se confirma, no interpretar."
        ],
        [
            "Calidad de muestra",
            "Calidad de señal",
            data["calidad_senal"],
            "Si es mala, repetir medición."
        ],
        [
            "Entrada ML",
            "Edad materna",
            f"{ml_input['Age']} años",
            "Usado por el modelo ML de riesgo materno."
        ],
        [
            "Entrada ML",
            "Presión arterial",
            f"{ml_input['SystolicBP']}/{ml_input['DiastolicBP']} mmHg",
            "Usado por el modelo ML de riesgo materno."
        ],
        [
            "Entrada ML",
            "Glucosa",
            f"{ml_input['BS']:.2f} mmol/L",
            "Usado por el modelo ML de riesgo materno."
        ],
        [
            "Entrada ML",
            "Temperatura",
            f"{data['temperatura_c']:.1f} °C / {ml_input['BodyTemp']:.1f} °F",
            "Usado por el modelo ML de riesgo materno."
        ],
        [
            "Entrada ML",
            "Frecuencia cardiaca materna",
            f"{ml_input['HeartRate']} lpm",
            "Usado por el modelo ML de riesgo materno."
        ],
        [
            "Contexto clínico",
            "FHR basal",
            f"{data['fhr_baseline']} lpm" if data["fhr_disponible"] else "No registrada",
            "No entra al modelo ML de riesgo materno; sirve como alerta clínica."
        ],
        [
            "Contexto clínico",
            "Sangrado",
            "Sí" if data["sangrado"] else "No",
            "Alerta clínica de seguridad."
        ],
        [
            "Contexto clínico",
            "Pérdida de líquido",
            "Sí" if data["perdida_liquido"] else "No",
            "Alerta clínica de seguridad."
        ],
        [
            "Contexto clínico",
            "Movimientos fetales disminuidos",
            "Sí" if data["movimientos_disminuidos"] else "No",
            "Alerta clínica de seguridad."
        ],
        [
            "Resultado",
            "Predicción ML",
            pred_label,
            "Salida del modelo ML de riesgo materno."
        ],
        [
            "Resultado",
            "Prioridad final de atención",
            f"{priority_percent:.0f}% - {final_level}",
            "Decisión final del sistema considerando ML + alertas clínicas."
        ]
    ]

    return pd.DataFrame(
        rows,
        columns=["Sección", "Dato", "Valor", "Interpretación"]
    )


def make_quick_report_text(final_level, pred_label, action, motives, priority_percent):
    report_lines = [
        "RESUMEN DE EVALUACIÓN INICIAL ML - MATERNIA",
        "=" * 52,
        f"Resultado final: {final_level}",
        f"Prioridad final de atención: {priority_percent:.0f}%",
        f"Predicción ML de riesgo materno: {pred_label}",
        f"Acción: {action}",
        "",
        "Motivos:"
    ]

    for motive in motives:
        report_lines.append(f"- {motive}")

    report_lines.append("")
    report_lines.append(
        "Nota: No es diagnóstico definitivo. La decisión final debe ser clínica."
    )

    return "\n".join(report_lines)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("👶 MaternIA")
st.sidebar.success("Modelos cargados")

st.sidebar.write("Modelo ML de riesgo materno:")
st.sidebar.code("maternIA_quick_triage_model.pkl")

st.sidebar.write("Modelo fetal CTG:")
st.sidebar.code("maternIA_fetal_health_model.pkl")

st.sidebar.divider()

st.sidebar.subheader("Flujo")
st.sidebar.write("**Posta:** ML de riesgo materno + seguridad clínica.")
st.sidebar.write("**CTG procesado:** ML fetal.")
st.sidebar.write("**Información:** criterios, variables y limitaciones.")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>👶 MaternIA</h1>
        <p>Apoyo al triaje fetal y obstétrico con Machine Learning, validación de muestra y resultados accionables.</p>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        "<div class='metric-card'><div class='metric-label'>Triaje inicial</div><div class='metric-value'>Riesgo materno</div></div>",
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        "<div class='metric-card'><div class='metric-label'>CTG procesado</div><div class='metric-value'>ML fetal</div></div>",
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        "<div class='metric-card'><div class='metric-label'>Modelo fetal</div><div class='metric-value'>LightGBM</div></div>",
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        "<div class='metric-card'><div class='metric-label'>Seguridad</div><div class='metric-value'>Valida muestra</div></div>",
        unsafe_allow_html=True
    )

st.warning(
    "Prototipo académico. No reemplaza diagnóstico médico. "
    "La decisión final debe ser tomada por personal de salud."
)

st.subheader("Cómo se debe usar MaternIA")

use_df = pd.DataFrame({
    "Modo": [
        "Evaluar paciente",
        "CTG procesado",
        "Información"
    ],
    "Usuario": [
        "Personal de posta",
        "Personal con archivo CTG",
        "Docente / personal de salud"
    ],
    "Usa modelo ML": [
        "Sí",
        "Sí",
        "No directamente"
    ],
    "Para qué sirve": [
        "Estimar riesgo materno-obstétrico inicial con datos simples",
        "Predecir Normal / Sospechoso / Patológico con CTG procesado",
        "Consultar criterios, variables técnicas y limitaciones"
    ]
})

st.dataframe(
    use_df,
    use_container_width=True,
    hide_index=True
)

st.markdown(
    """
    <div class="important-box">
    <b>Importante:</b> la evaluación inicial usa un modelo ML de riesgo materno con datos simples de posta.
    El modelo fetal de 21 variables solo debe usarse cuando los datos provienen de un cardiotocógrafo
    o software CTG procesado.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS PRINCIPALES
# ============================================================

tab_quick, tab_ctg, tab_info = st.tabs([
    "Evaluar paciente",
    "CTG procesado",
    "Información"
])


# ============================================================
# TAB 1: EVALUAR PACIENTE
# ============================================================

with tab_quick:
    st.header("Evaluación inicial con modelo ML de riesgo materno")

    st.markdown(
        """
        <div class="info-box">
        Este módulo usa un modelo de Machine Learning entrenado con el dataset Maternal Health Risk.
        Estima el riesgo materno-obstétrico inicial usando variables simples que pueden registrarse en una posta médica.
        Además aplica una capa de seguridad: si existe una alerta clínica crítica, la prioridad final puede subir
        aunque la probabilidad del modelo sea baja.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Calidad de muestra obligatoria")

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        edad_gestacional_registrada = st.checkbox(
            "Edad gestacional registrada",
            value=True
        )

    with q2:
        latido_confirmado = st.checkbox(
            "Latido fetal confirmado",
            value=True
        )

    with q3:
        calidad_senal = st.selectbox(
            "Calidad de señal",
            ["Buena", "Dudosa", "Mala / no confiable"]
        )

    with q4:
        motivo_evaluacion = st.text_input(
            "Motivo de evaluación",
            value="Control o sospecha de riesgo"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Entradas del modelo ML")

        edad_materna = st.number_input(
            "Edad materna (años)",
            min_value=10,
            max_value=70,
            value=28,
            step=1
        )

        sistolica = st.number_input(
            "Presión sistólica (mmHg)",
            min_value=70,
            max_value=240,
            value=120,
            step=1
        )

        diastolica = st.number_input(
            "Presión diastólica (mmHg)",
            min_value=40,
            max_value=150,
            value=80,
            step=1
        )

        frecuencia_materna = st.number_input(
            "Frecuencia cardiaca materna (lpm)",
            min_value=40,
            max_value=180,
            value=80,
            step=1
        )

    with col2:
        st.subheader("Glucosa y temperatura")

        unidad_glucosa = st.selectbox(
            "Unidad de glucosa",
            ["mg/dL", "mmol/L"]
        )

        if unidad_glucosa == "mg/dL":
            glucosa_valor = st.number_input(
                "Glucosa",
                min_value=40.0,
                max_value=500.0,
                value=110.0,
                step=1.0
            )
            glucosa_mmol = mgdl_to_mmol_l(glucosa_valor)
        else:
            glucosa_valor = st.number_input(
                "Glucosa",
                min_value=2.0,
                max_value=30.0,
                value=6.1,
                step=0.1
            )
            glucosa_mmol = glucosa_valor

        temperatura_c = st.number_input(
            "Temperatura materna (°C)",
            min_value=34.0,
            max_value=42.0,
            value=36.8,
            step=0.1
        )

        temperatura_f = celsius_to_fahrenheit(temperatura_c)

        fhr_disponible = st.checkbox(
            "Se pudo medir FHR basal",
            value=True
        )

        if fhr_disponible:
            fhr_baseline = st.number_input(
                "FHR basal (lpm)",
                min_value=60,
                max_value=220,
                value=140,
                step=1
            )
        else:
            fhr_baseline = np.nan

    with col3:
        st.subheader("Alertas clínicas de contexto")

        movimientos_disminuidos = st.checkbox("Disminución de movimientos fetales")
        sangrado = st.checkbox("Sangrado vaginal")
        perdida_liquido = st.checkbox("Pérdida de líquido")
        dolor_intenso = st.checkbox("Dolor abdominal intenso")
        cefalea_vision = st.checkbox("Cefalea intensa, visión borrosa o edema marcado")
        convulsiones = st.checkbox("Convulsiones o episodio neurológico severo")

        contracciones = st.selectbox(
            "Contracciones",
            ["No", "Irregulares", "Frecuentes / muy dolorosas"]
        )

        desaceleraciones = st.selectbox(
            "Desaceleraciones observadas o escuchadas",
            [
                "No",
                "Sí, leves o aisladas",
                "Sí, prolongadas o repetidas",
                "No se pudo evaluar"
            ]
        )

    quick_data = {
        "edad_gestacional_registrada": edad_gestacional_registrada,
        "motivo_evaluacion": motivo_evaluacion,
        "latido_confirmado": latido_confirmado,
        "calidad_senal": calidad_senal,
        "sistolica": sistolica,
        "diastolica": diastolica,
        "temperatura_c": temperatura_c,
        "movimientos_disminuidos": movimientos_disminuidos,
        "sangrado": sangrado,
        "perdida_liquido": perdida_liquido,
        "dolor_intenso": dolor_intenso,
        "cefalea_vision": cefalea_vision,
        "convulsiones": convulsiones,
        "fhr_disponible": fhr_disponible,
        "fhr_baseline": fhr_baseline,
        "contracciones": contracciones,
        "desaceleraciones": desaceleraciones
    }

    ml_input = {
        "Age": float(edad_materna),
        "SystolicBP": float(sistolica),
        "DiastolicBP": float(diastolica),
        "BS": float(glucosa_mmol),
        "BodyTemp": float(temperatura_f),
        "HeartRate": float(frecuencia_materna)
    }

    st.divider()

    if st.button("Evaluar paciente", type="primary"):
        blocking, sample_warnings = quick_safety_gate(quick_data)

        if blocking:
            render_action_card(
                title="⚪ No interpretable: repetir medición",
                action="No emitir clasificación con esta muestra. Repetir medición y completar datos mínimos.",
                motives=blocking,
                level="No interpretable"
            )

            show_final_priority(
                final_level="No interpretable",
                priority_percent=0.0,
                source_text="La muestra no cumple criterios mínimos de interpretación."
            )

        else:
            input_df = pd.DataFrame(
                [ml_input],
                columns=QUICK_FEATURE_COLUMNS
            )

            pred = normalize_int(
                quick_model.predict(input_df)[0]
            )

            pred_label = QUICK_CLASS_NAMES.get(pred, str(pred))

            probabilities = None
            classes = None
            model_confidence_percent = 0.0

            if hasattr(quick_model, "predict_proba"):
                probabilities = quick_model.predict_proba(input_df)[0]
                classes = get_model_classes(
                    quick_model,
                    [0, 1, 2]
                )
                model_confidence_percent = float(np.max(probabilities) * 100)

            alerts, cautions = clinical_alerts_from_context(quick_data)
            all_cautions = sample_warnings + cautions

            title, action, motives, final_level, has_clinical_alerts = quick_action_from_prediction(
                pred_label,
                alerts,
                all_cautions
            )

            priority_percent = get_final_priority_percent(
                final_level=final_level,
                has_clinical_alerts=has_clinical_alerts,
                model_confidence_percent=model_confidence_percent
            )

            render_action_card(
                title=title,
                action=action,
                motives=motives,
                level=final_level
            )

            if has_clinical_alerts:
                source_text = (
                    "Prioridad elevada por alerta clínica de seguridad. "
                    "No es la probabilidad del modelo ML."
                )
            else:
                source_text = (
                    "Prioridad basada en la predicción del modelo ML de riesgo materno "
                    "y en los criterios de seguridad."
                )

            show_final_priority(
                final_level=final_level,
                priority_percent=priority_percent,
                source_text=source_text
            )

            if probabilities is not None:
                show_probabilities(
                    probabilities,
                    classes,
                    QUICK_CLASS_NAMES
                )

            with st.expander("Ver resumen legible de la evaluación", expanded=True):
                summary_df = build_quick_summary_table(
                    quick_data,
                    ml_input,
                    pred_label,
                    final_level,
                    priority_percent
                )

                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True
                )

                report_text = make_quick_report_text(
                    final_level=final_level,
                    pred_label=pred_label,
                    action=action,
                    motives=motives,
                    priority_percent=priority_percent
                )

                st.download_button(
                    "Descargar resumen",
                    report_text.encode("utf-8"),
                    file_name="resumen_evaluacion_inicial_ml_maternia.txt",
                    mime="text/plain"
                )


# ============================================================
# TAB 2: CTG PROCESADO
# ============================================================

with tab_ctg:
    st.header("Cargar CTG procesado")

    st.markdown(
        """
        <div class="info-box">
        Este modo usa el modelo fetal LightGBM. Debe recibir un CSV con las 21 variables técnicas
        ya procesadas desde un cardiotocógrafo o software de análisis CTG.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "No uses este modo si solo tienes datos básicos de posta. "
        "Para eso usa Evaluar paciente."
    )

    with st.expander("Columnas requeridas"):
        columns_df = pd.DataFrame({"Variable requerida": FETAL_FEATURE_COLUMNS})
        st.dataframe(
            columns_df,
            use_container_width=True,
            hide_index=True
        )

    uploaded_csv = st.file_uploader(
        "Subir archivo CSV con CTG procesado",
        type=["csv"]
    )

    if uploaded_csv is not None:
        try:
            raw_df = pd.read_csv(uploaded_csv)

            st.subheader("Vista previa")
            st.dataframe(
                raw_df.head(),
                use_container_width=True
            )

            if "fetal_health" in raw_df.columns:
                raw_df = raw_df.drop(columns=["fetal_health"])

            if "NSP" in raw_df.columns:
                raw_df = raw_df.drop(columns=["NSP"])

            missing = [
                col for col in FETAL_FEATURE_COLUMNS
                if col not in raw_df.columns
            ]

            if missing:
                st.error(
                    "Faltan columnas: " + ", ".join(missing)
                )
            else:
                clean_df = raw_df[FETAL_FEATURE_COLUMNS].copy()

                for col in FETAL_FEATURE_COLUMNS:
                    clean_df[col] = pd.to_numeric(
                        clean_df[col],
                        errors="coerce"
                    )

                if clean_df.isnull().any().any():
                    st.error(
                        "Hay valores vacíos o no numéricos en el CSV."
                    )
                else:
                    result_df = clean_df.copy()

                    preds = fetal_model.predict(clean_df)
                    preds = [normalize_int(pred) for pred in preds]

                    result_df["prediccion_num"] = preds
                    result_df["estado_fetal_predicho"] = result_df["prediccion_num"].map(
                        FETAL_CLASS_NAMES
                    )

                    if hasattr(fetal_model, "predict_proba"):
                        probabilities_fetal = fetal_model.predict_proba(clean_df)
                        fetal_classes = get_model_classes(
                            fetal_model,
                            [1, 2, 3]
                        )

                        for idx, cls in enumerate(fetal_classes):
                            cls = normalize_int(cls)
                            class_label = FETAL_CLASS_NAMES.get(cls, str(cls))
                            result_df[f"prob_{class_label}_porcentaje"] = np.round(
                                probabilities_fetal[:, idx] * 100,
                                2
                            )

                    st.subheader("Resultados ML fetal")
                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    resumen = result_df["estado_fetal_predicho"].value_counts().reset_index()
                    resumen.columns = ["Estado fetal", "Cantidad"]

                    st.subheader("Resumen")
                    st.dataframe(
                        resumen,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.bar_chart(
                        resumen.set_index("Estado fetal")
                    )

                    st.download_button(
                        "Descargar resultados",
                        result_df.to_csv(index=False).encode("utf-8"),
                        file_name="resultados_ctg_maternia.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error("No se pudo procesar el archivo.")
            st.exception(e)


# ============================================================
# TAB 3: INFORMACION
# ============================================================

with tab_info:
    st.header("Información del sistema")

    with st.expander("Modo avanzado académico", expanded=False):
        st.warning(
            "Este modo usa el modelo fetal con 21 variables CTG. "
            "No está diseñado para ingreso manual por personal no especializado."
        )

        st.subheader("Variables requeridas por el modelo fetal")

        variable_df = pd.DataFrame({
            "Variable": FETAL_FEATURE_COLUMNS
        })

        st.dataframe(
            variable_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Para predicción individual avanzada, crea un CSV con una sola fila "
            "y súbelo en la pestaña 'CTG procesado'."
        )

    with st.expander("Criterios de muestreo", expanded=False):
        st.markdown(
            """
            - Confirmar que el latido registrado sea fetal y no materno.
            - Confirmar que la señal sea clara.
            - Registrar edad gestacional.
            - Registrar motivo de evaluación.
            - Repetir medición si hay pérdida de señal o dudas.
            - No usar el sistema como diagnóstico definitivo.
            """
        )

        triage_rules = pd.DataFrame({
            "Criterio": [
                "Muestra completa y confiable",
                "Latido fetal no confirmado",
                "Señal mala/no confiable",
                "Predicción ML de riesgo bajo sin alertas clínicas",
                "Predicción ML de riesgo medio",
                "Predicción ML de riesgo alto",
                "Sangrado, pérdida de líquido, fiebre, PA alta o FHR fuera de rango"
            ],
            "Acción": [
                "Interpretar con ML",
                "No interpretable / repetir medición",
                "No interpretable / repetir medición",
                "Continuar control",
                "Repetir medición o consultar",
                "Derivar o contactar obstetra",
                "Priorizar seguridad clínica aunque el ML sea bajo"
            ]
        })

        st.dataframe(
            triage_rules,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("Limitaciones del modelo", expanded=False):
        limitations_df = pd.DataFrame({
            "Limitación": [
                "Dos modelos distintos",
                "Modelo ML de riesgo materno no es diagnóstico fetal",
                "Modelo fetal requiere CTG procesado",
                "Dataset público",
                "No reemplaza al médico",
                "No validado localmente",
                "Supervisión humana obligatoria",
                "Probabilidad ML no equivale a prioridad final"
            ],
            "Qué significa": [
                "El triaje inicial estima riesgo materno-obstétrico; el modelo CTG estima estado fetal.",
                "Usa signos vitales maternos simples; no reemplaza una cardiotocografía.",
                "Necesita las 21 variables técnicas generadas por CTG/software.",
                "Fue entrenado con datasets públicos, no con datos locales peruanos.",
                "El resultado es apoyo al triaje, no diagnóstico definitivo.",
                "Debe evaluarse con datos locales antes de usarse en atención real.",
                "La decisión final debe tomarla personal de salud.",
                "Si existe una alerta clínica, la prioridad final puede ser alta aunque el modelo ML tenga baja probabilidad."
            ]
        })

        st.dataframe(
            limitations_df,
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "Si hay sangrado, pérdida de líquido, movimientos fetales disminuidos, fiebre, "
            "presión alta o FHR fuera de rango, la seguridad clínica está por encima del resultado del modelo."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MaternIA es un prototipo académico de apoyo al triaje fetal y obstétrico. "
    "No debe usarse como diagnóstico médico real sin validación clínica local."
)