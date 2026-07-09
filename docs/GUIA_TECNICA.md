# Guía técnica de MaternIA

## Instalación

```bash
pip install -r requirements.txt
```

## Entrenamiento

```bash
python train_maternia_integral.py
```

## Ejecución

```bash
python -m streamlit run app_streamlit_maternia.py
```

## Archivos generados

- `models/maternIA_maternal_risk_model.pkl`
- `models/maternIA_fetal_health_model.pkl`
- `models/maternIA_fetal_mlp_neural_network.pkl`
- `reports/*.csv`
- `reports/*.png`
- `reports/*.json`
- `reports/*_classification_report*.txt`

## Componentes técnicos

- Preprocesamiento: conversión numérica, normalización de etiquetas y estandarización de columnas CTG.
- Balanceo de clases: `SMOTE` durante el entrenamiento.
- Modelos comparados: regresión logística, random forest, gradient boosting, LightGBM y red neuronal `MLPClassifier`.
- Validación: división estratificada 80/20 y búsqueda con validación cruzada para LightGBM cuando está disponible.
- Transparencia: reportes de métricas, matrices de confusión, importancia de variables y servicio cognitivo de triaje.

## Advertencia ética

MaternIA es un prototipo académico de apoyo al triaje preventivo. No reemplaza diagnóstico médico ni evaluación profesional.
