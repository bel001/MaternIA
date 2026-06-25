# Guion breve de exposición

MaternIA Integral es un prototipo académico de apoyo al triaje materno-fetal para postas médicas rurales. La versión final usa dos modelos separados. El primero evalúa riesgo materno con variables como edad, presión arterial, glucosa, temperatura y frecuencia cardiaca. El segundo evalúa el estado fetal mediante variables cardiotocográficas procesadas.

No fusionamos los datasets porque no pertenecen a las mismas pacientes. En cambio, entrenamos modelos independientes y luego integramos sus resultados mediante una capa de decisión final. Esto permite considerar variables maternas importantes sin alterar incorrectamente el modelo fetal.

La métrica más importante del modelo materno es el recall de riesgo alto, porque interesa detectar casos maternos críticos. En el modelo fetal se prioriza el recall patológico, porque el error más grave sería no identificar un caso fetal de alto riesgo.

La app tiene un modo posta, un modo CTG procesado, un modo integral y un modo académico. El sistema no diagnostica; solo apoya la priorización inicial y la toma de decisiones de referencia.
