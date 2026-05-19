# MaternIA

MaternIA es un prototipo academico desarrollado en Streamlit para apoyar el
triaje materno-obstetrico mediante modelos de Machine Learning y reglas de
seguridad clinica. La aplicacion no emite diagnosticos definitivos: organiza
senales de riesgo, muestra probabilidades del modelo y ayuda a priorizar la
atencion cuando existen alertas clinicas.

## Que hace

- Evalua riesgo materno-obstetrico inicial con datos simples de posta medica:
  edad, presion arterial, glucosa, temperatura y frecuencia cardiaca materna.
- Aplica una capa de seguridad clinica para elevar la prioridad si hay alertas
  como sangrado, perdida de liquido, fiebre, presion alta, convulsiones o FHR
  fuera de rango.
- Procesa archivos CSV con variables CTG ya calculadas para clasificar estado
  fetal como Normal, Sospechoso o Patologico.
- Permite descargar resumenes de evaluacion y resultados procesados.

## Archivos incluidos

- `app_streamlit_maternia.py`: aplicacion principal de Streamlit.
- `requirements.txt`: dependencias necesarias para ejecutar la aplicacion.

Los modelos entrenados no estan incluidos en este repositorio. Para ejecutar la
app se deben colocar estos archivos en la misma carpeta que el `.py`:

- `maternIA_quick_triage_model.pkl`
- `maternIA_fetal_health_model.pkl`

## Ejecucion local

```bash
pip install -r requirements.txt
streamlit run app_streamlit_maternia.py
```

## Alcance

Este proyecto esta orientado a demostracion academica y validacion conceptual.
Antes de cualquier uso clinico real debe validarse con datos locales,
supervision de personal de salud y protocolos institucionales.
