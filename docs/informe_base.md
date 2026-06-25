# Informe base del proyecto

## Título

MaternIA Integral: Sistema de apoyo al triaje materno-fetal preventivo para postas médicas rurales.

## Problema

En postas rurales puede existir limitación de especialistas y herramientas de apoyo para priorizar casos materno-fetales. MaternIA busca organizar señales de riesgo y apoyar la priorización inicial.

## Objetivo general

Desarrollar un prototipo académico que integre un modelo de riesgo materno y un modelo de estado fetal para apoyar el triaje preventivo.

## Metodología

Se entrenan dos modelos independientes:

1. Modelo materno con Maternal Health Risk.
2. Modelo fetal con UCI Cardiotocography / Fetal Health.

Ambos se evalúan con división 80/20 estratificada, validación cruzada y métricas de clasificación.

## Métricas principales

- Modelo materno: Recall de Riesgo Alto.
- Modelo fetal: Recall Patológico.

## Integración

La salida final se obtiene mediante reglas de decisión transparentes que combinan riesgo materno, estado fetal, edad gestacional y signos de alarma.

## Conclusión

El proyecto demuestra una arquitectura más completa que considera tanto variables maternas importantes como información fetal cardiotocográfica, manteniendo separación metodológica entre datasets.
