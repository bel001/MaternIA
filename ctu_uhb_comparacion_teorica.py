from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)

texto = """# CTU-UHB / CTU-CHB como extensión avanzada

En esta versión final de MaternIA, los dos modelos funcionales son:

1. Modelo materno con Maternal Health Risk.
2. Modelo fetal con UCI Cardiotocography / Fetal Health.

CTU-UHB / CTU-CHB se conserva como una extensión avanzada, porque no es una tabla simple lista para entrenar: contiene señales cardiotocográficas reales y requiere procesamiento de series temporales.

## Ruta futura

1. Descargar señales CTU-UHB desde PhysioNet.
2. Leer registros de frecuencia cardiaca fetal y contracciones uterinas.
3. Limpiar artefactos y segmentos inválidos.
4. Extraer características estadísticas y clínicas por ventanas.
5. Entrenar un tercer modelo basado en señal cruda/procesada.
6. Comparar con el modelo tabular fetal actual.

## Justificación

No se usa CTU-UHB como modelo funcional dentro de esta entrega para evitar un procesamiento superficial de señales. Se documenta como evolución técnica seria del proyecto.
"""

out = DOCS_DIR / "ctu_uhb_modulo_avanzado.md"
out.write_text(texto, encoding="utf-8")
print(f"Documento generado: {out}")
