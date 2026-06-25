# MaternIA Integral: Triaje materno-fetal preventivo

MaternIA Integral es un prototipo académico en Streamlit para apoyar el triaje preventivo en postas médicas rurales. La versión final usa **dos modelos separados** y una capa de decisión integrada.

## Modelos funcionales

### 1. Modelo de riesgo materno

Dataset: **Maternal Health Risk**.

Variables:

- `Age`
- `SystolicBP`
- `DiastolicBP`
- `BS`
- `BodyTemp`
- `HeartRate`

Salida:

- Bajo
- Medio
- Alto

### 2. Modelo de estado fetal

Dataset: **UCI Cardiotocography / Fetal Health Classification**.

Variables:

- 21 variables cardiotocográficas procesadas.

Salida:

- Normal
- Sospechoso
- Patológico

## Integración

Los datasets **no se fusionan** porque no pertenecen a las mismas pacientes. El sistema entrena modelos separados y luego integra los resultados con reglas de triaje:

- Riesgo materno alto → prioridad alta.
- Estado fetal patológico → referencia urgente.
- CTG sospechoso + riesgo materno medio/alto → prioridad alta.
- Signos de alarma clínicos → prioridad alta o referencia urgente.

## Instalación

```bash
cd MaternIA_integral_materno_fetal_final
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entrenamiento

```bash
python train_maternia_integral.py
```

Esto genera:

```text
models/maternIA_maternal_risk_model.pkl
models/maternIA_fetal_health_model.pkl
reports/maternal_*.csv/png/txt/json
reports/fetal_*.csv/png/txt/json
reports/integrated_training_summary.json
```

## Ejecución

```bash
python -m streamlit run app_streamlit_maternia.py
```

Luego abre:

```text
http://localhost:8501
```

## Modos de la app

1. **Evaluación integral**: combina modelo materno + signos clínicos + CTG opcional.
2. **Modo posta**: usa datos simples y modelo materno.
3. **CTG procesado**: carga CSV con 21 variables y predice estado fetal.
4. **Avanzado académico**: ingreso manual de variables para exposición.
5. **Reportes**: muestra métricas y matrices de confusión.

## Advertencia ética

MaternIA es un prototipo académico. No emite diagnóstico médico ni reemplaza la evaluación clínica profesional.
