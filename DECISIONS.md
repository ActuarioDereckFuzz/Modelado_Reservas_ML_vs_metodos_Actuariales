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
| D10 | Sustituir `GLM_FEATURES` / `TREE_FEATURES` (con `delta_1m`, `mean_delta_3m`, `std_delta_3m`, flags de historia) por el set simplificado de `build_ml_features` (`snapshot_dev_month`, `development_fraction`, `observed_amount`, `accident_year`, `accident_month_sin/cos`) | El set extendido exige imputación en `dev_month` 0–2 (sin historial para calcular deltas) y, con solo 12 cohortes maduras mínimas por escenario, 10 variables aumentan el riesgo de sobreajuste sin evidencia de que aporten señal adicional. El set reducido cubre edad, fracción de desarrollo, nivel observado y estacionalidad sin necesitar imputación ni flags auxiliares | ✅ |
| D11 | Sustituir XGBoost por `HistGradientBoostingRegressor` (sklearn) y añadir Ridge a la familia de Machine Learning | Evita una dependencia externa (`xgboost`) fuera del entorno `Diplomado_ML` del equipo. `HistGradientBoostingRegressor` maneja valores faltantes de forma nativa y da un desempeño equivalente en este volumen de datos. Ridge se agregó como referencia lineal regularizada, útil para diagnosticar si la no linealidad de los árboles aporta valor real frente a un modelo simple | ✅ |
| D12 | Sustituir `DEV_BANDS` fijas (0-5, 6-11, 12-23...) por bandas dinámicas de 6 meses calculadas en el notebook 05 | Las bandas fijas asumen cobertura pareja en todas las edades, pero las primeras fechas de valuación del backtesting (2020-01 a ~2020-11) tienen pocas o ninguna cohorte en edades altas. Bandas dinámicas calculadas sobre el rango de edades realmente evaluado evitan reportar una métrica sobre una banda con muestra insuficiente | ✅ |
| D13 | Usar MAE como métrica primaria de selección en vez de MAPE, y omitir la regla de competitividad relativa de 5 % (`RELATIVE_SELECTION_THRESHOLD`) definida en el punto 3 | El escenario mixto produce una reserva esperada cercana a cero en algunos cortes de valuación, lo que vuelve el error porcentual (MAPE) inestable o indefinido cuando el target se acerca a cero. MAE mantiene las mismas unidades monetarias en los tres escenarios y es comparable directamente sin ese riesgo de explosión numérica. El umbral de competitividad relativa se reemplazó por una comparación directa de MAE, conservando de forma informal el mismo orden de desempate (sesgo → bandas → estabilidad → simplicidad) sobre el ranking resultante | ✅ |

> **Nota:** D10–D13 reconcilian lo congelado en `src/experiment/config.py` (notebook 3)
> con lo que `src/models` y el notebook 05 implementaron realmente. Discutido y
> acordado con el equipo; no se detectó fuga de información en las features
> efectivamente usadas, así que no invalida los resultados reportados en
> `PROJECT_STATE.md`.
## Decisiones pendientes

- Horizonte definitivo `dev_M`.
- Período definitivo de simulación.
- Fechas de valuación.
- Definición definitiva del target.
- Diseño final del backtesting.
- Métricas principales de comparación.
- Actualizar Notebook 3 para indicar que algunas features no están vigenetes en pasos posteriores.