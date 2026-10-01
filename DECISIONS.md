# Decisions — Loss Incurred

| ID | Decisión | Motivo | Estado |
|---|---|---|---|
| D01 | Utilizar datos mensuales | Analizar desarrollo y estacionalidad con mayor granularidad | ✅ |
| D02 | Simular tres escenarios: creciente, decreciente y mixto | Comparar métodos bajo diferentes patrones de desarrollo | ✅ |
| D03 | Incorporar tendencia, estacionalidad y ruido | Generar datos menos triviales y más realistas | ✅ |
| D04 | Usar métodos tradicionales como benchmark | Comparar el valor adicional de modelos estadísticos | ✅ |
| D05 | Utilizar GLM Gamma con enlace log | Adecuado para una respuesta continua y positiva | ✅ |
| D06 | Evaluar los modelos sobre las mismas claves | Garantizar comparabilidad | ✅ |
| D07 | Backtesting estrictamente temporal | Evitar leakage | ✅ |
| D08 | Revisar el backtesting inicial | No evalúa adecuadamente edades jóvenes | 🔄 |
| D09 | Revisar `dev_M` y período histórico | Deben ser compatibles con el nuevo backtesting | 🔄 |

## Decisiones pendientes

- Horizonte definitivo `dev_M`.
- Período definitivo de simulación.
- Fechas de valuación.
- Definición definitiva del target.
- Diseño final del backtesting.
- Métricas principales de comparación.