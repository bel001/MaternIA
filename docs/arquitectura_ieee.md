# Arquitectura IEEE resumida

## 1. Capa de datos

- Maternal Health Risk: variables maternas simples.
- UCI Cardiotocography / Fetal Health: variables CTG procesadas.

## 2. Capa de preprocesamiento

- Validación de columnas.
- Conversión numérica.
- División 80/20 estratificada.
- Manejo de desbalance con SMOTE.

## 3. Capa de modelos

- Modelo materno: clasifica riesgo Bajo, Medio o Alto.
- Modelo fetal: clasifica CTG Normal, Sospechoso o Patológico.

## 4. Capa de evaluación

- Accuracy.
- Balanced Accuracy.
- Precision Macro.
- Recall Macro.
- F1 Macro.
- Recall de clase crítica.
- Matriz de confusión.
- Importancia de variables.

## 5. Capa de decisión integrada

Integra riesgo materno, estado fetal, edad gestacional y signos de alarma.

## 6. Capa de presentación

Aplicación Streamlit con modos de uso diferenciados.
