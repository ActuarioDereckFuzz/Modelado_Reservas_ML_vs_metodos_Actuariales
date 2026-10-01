# Project State — Loss Incurred

**Última actualización:** 2026-10-01

## Estado general

🟢 **En desarrollo — Fases 01 y 02 completadas**

## Fase actual

**03 — Triángulos y métodos actuariales**

## Completado

### Fase 01 — Diseño y simulación ✅

- Tres escenarios: creciente, decreciente y mixto.
- Simulación mensual `2015-01` a `2025-12`.
- `dev_M = 60 / 48 / 48`.
- Tendencia, estacionalidad y ruido incorporados.
- Positividad, dimensiones, calendario y madurez validados.
- Reproducibilidad validada.
- `LI_M = Ultimate`.
- `src/simulation` validado.
- `30 tests` superados.
- Notebook `01_diseno_simulacion.ipynb` terminado.

### Fase 02 — Validación y exploración ✅

- Estructura de las tres bases validada.
- Períodos de ocurrencia y desarrollo revisados.
- Triángulos acumulados e incrementales construidos y revisados.
- Diagonal más reciente obtenida.
- Tendencia y estacionalidad analizadas.
- Factores edad-a-edad calculados.
- Tabla completa de factores incorporada.
- Heatmap incremental revisado excluyendo `dev = 0`.
- Comportamiento esperado de los tres escenarios corroborado.
- Requisitos 4–7 verificados explícitamente.
- Notebook `02_validacion_exploracion_final.ipynb` finalizado y preparado para ejecución reproducible.

## Decisiones vigentes

- Mantener `dev_M = 60 / 48 / 48`.
- Mantener los tres escenarios originales.
- No modificar la lógica de simulación de este experimento.
- Trabajar con periodicidad mensual.
- Mantener separación entre simulación, validación, modelación y backtesting.

## Próxima fase

### 03 — Triángulos y métodos actuariales

Objetivos principales:

- definir formalmente los métodos tradicionales de desarrollo;
- implementar alternativas de factores;
- proyectar valores futuros;
- estandarizar predicciones;
- preparar benchmarks para compararlos posteriormente con los modelos estadísticos.

## Estado por fase

| Fase | Estado |
|---|---|
| 00 — Roadmap y estado | 🟢 Continuo |
| 01 — Diseño y simulación | ✅ Completada |
| 02 — Validación y exploración | ✅ Completada |
| 03 — Triángulos y métodos actuariales | 🟡 Siguiente |
| 04 — Modelación estadística / ML | ⚪ Pendiente |
| 05 — Backtesting | ⚪ Pendiente |
| 06 — Comparación y resultados | ⚪ Pendiente |
| 07 — Documentación final | ⚪ Pendiente |

## Próximo objetivo

Construir los métodos actuariales de referencia que funcionarán como benchmark para los modelos estadísticos y de ML.