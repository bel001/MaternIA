# Datos del proyecto MaternIA

Este proyecto usa dos datasets funcionales:

1. **Maternal Health Risk**: para el modelo de riesgo materno. Variables: `Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`. Salida: `RiskLevel`.
2. **UCI Cardiotocography / Fetal Health**: para el modelo fetal CTG. Variables cardiotocográficas procesadas. Salida: Normal, Sospechoso o Patológico.

Los datasets se descargan automáticamente desde UCI con `ucimlrepo` cuando ejecutas `train_maternia_integral.py`.

Archivos de ejemplo incluidos:

- `ejemplo_materno.csv`: datos simples para probar el modelo materno.
- `ejemplo_ctg_procesado.csv`: datos CTG procesados para probar el modelo fetal.

Opcionalmente puedes colocar archivos locales:

- `data/maternal_health_risk.csv`
- `data/fetal_health.csv`

Si existen, el script intentará usarlos primero.
