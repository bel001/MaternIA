# Comparación de datasets usados en MaternIA Integral

| Aspecto | Maternal Health Risk | UCI Cardiotocography / Fetal Health |
|---|---|---|
| Rol | Modelo materno | Modelo fetal |
| Entrada | Edad, presión, glucosa, temperatura, FC materna | 21 variables CTG procesadas |
| Salida | Bajo, Medio, Alto | Normal, Sospechoso, Patológico |
| Usuario principal | Posta médica / triaje inicial | Personal con archivo CTG procesado |
| Integración | Capa de decisión final | Capa de decisión final |

## Decisión metodológica

Los datasets no se unen en una sola tabla porque no pertenecen a las mismas pacientes. La forma correcta es entrenar dos modelos independientes y combinar sus salidas mediante reglas transparentes de triaje.
