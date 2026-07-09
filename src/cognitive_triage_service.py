def _normalize_label(value):
    return str(value or "").strip().lower()


def cognitive_triage_service(maternal_prediction, fetal_prediction, warning_signs=None):
    warning_signs = warning_signs or []
    maternal = _normalize_label(maternal_prediction)
    fetal = _normalize_label(fetal_prediction)

    explanations = []
    priority = "BAJA"

    if maternal in {"alto", "high risk", "high"}:
        priority = "ALTA"
        explanations.append("El modelo materno detecta riesgo alto.")

    if fetal in {"patologico", "patológico", "pathological"}:
        priority = "REFERENCIA URGENTE"
        explanations.append("El modelo fetal detecta estado patológico.")

    if fetal in {"sospechoso", "suspect"} and maternal in {"medio", "alto", "mid risk", "high risk", "medium", "high"}:
        if priority != "REFERENCIA URGENTE":
            priority = "ALTA"
        explanations.append("Existe combinación de riesgo materno y estado fetal sospechoso.")

    if warning_signs:
        if priority != "REFERENCIA URGENTE":
            priority = "ALTA"
        explanations.append("Se reportan signos clínicos de alarma.")
        explanations.extend(warning_signs)

    if not explanations:
        explanations.append("No se detectan condiciones críticas según las reglas del prototipo.")

    return {
        "priority": priority,
        "explanations": explanations,
        "ethical_warning": "Este resultado es un apoyo académico y no reemplaza evaluación médica profesional.",
    }
