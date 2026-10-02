# Project State — Loss Incurred

**Última actualización:** 2026-10-01

## Estado general

🟢 **En desarrollo — Fases 01, 02 y 03 completadas**

## Fase actual

**04 — Backtesting | Requisitos 11–12**

## Completado

### 01 — Diseño y simulación ✅

- Tres escenarios simulados:
  - creciente;
  - decreciente;
  - mixto.
- Periodicidad mensual.
- `dev_M = 60 / 48 / 48`.
- Tendencia, estacionalidad y ruido incorporados.
- Madurez, calendario, positividad y reproducibilidad validados.
- Notebook y módulo de simulación terminados.
- 30 tests superados.

### 02 — Validación y exploración ✅

- Estructura de las tres bases validada.
- Desarrollo observado revisado.
- Factores edad-a-edad analizados.
- Triángulos acumulados e incrementales construidos.
- Diagonal más reciente obtenida.
- Tendencia y estacionalidad exploradas.
- Comportamiento esperado de los tres escenarios corroborado.
- Notebook de validación y exploración terminado.

### 03 — Diseño del experimento ✅

- Protocolo temporal de evaluación definido.
- Fechas de valuación y elegibilidad formalizadas.
- Separación entre información disponible y futura establecida.
- Targets reales obtenidos en `dev_M`.
- Snapshots de entrenamiento y evaluación construidos.
- Validación de ausencia de leakage.
- Módulo `src.experiment` organizado.
- Notebook de diseño experimental terminado.
- 18 tests superados.

## Próxima fase

### 04 — Backtesting | Requisitos 11–12

Objetivos principales:

- ejecutar el backtesting histórico;
- entrenar los métodos únicamente con información disponible en cada valuación;
- generar predicciones comparables;
- revelar los targets únicamente cuando corresponda;
- almacenar resultados de forma estandarizada para todos los modelos.

## Estado por fase

| Fase | Estado |
|---|---|
| 00 — Roadmap y estado | 🟢 Continuo |
| 01 — Diseño y simulación | ✅ Completada |
| 02 — Validación y exploración | ✅ Completada |
| 03 — Diseño del experimento | ✅ Completada |
| 04 — Backtesting | 🟡 Siguiente |
| 05 — Evaluación y selección | ⚪ Pendiente |
| 06 — Reserva y reporte final | ⚪ Pendiente |

## Próximo objetivo

Implementar el backtesting completo sobre los tres escenarios siguiendo el protocolo temporal definido en la Fase 03.