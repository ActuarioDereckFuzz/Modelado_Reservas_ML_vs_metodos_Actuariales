# Project State — Loss Incurred

**Última actualización:** 2026-10-02

## Estado general

🟢 **En desarrollo — Backtesting completado**

## Fase actual

**05 — Evaluación y comparación de modelos**

## Completado

### 01 — Diseño y simulación ✅
- Escenarios creciente, decreciente y mixto.
- Simulación mensual reproducible.
- `dev_M = 60 / 48 / 48`.
- Tendencia, estacionalidad, ruido y madurez validados.

### 02 — Validación y exploración ✅
- Bases validadas.
- Triángulos acumulados e incrementales.
- Diagonal más reciente.
- Factores edad-a-edad.
- Tendencia y estacionalidad.
- Patrones esperados corroborados.

### 03 — Diseño del experimento ✅
- Protocolo temporal definido.
- Snapshots históricos.
- Separación training / prediction / target.
- Controles de leakage.
- Elegibilidad y revelación del target separadas.
- Tests unitarios e integración superados.

### 04 — Backtesting ✅
- Infraestructura de backtesting corregida y validada.
- Cohortes jóvenes incorporadas al esquema de predicción.
- Métodos de desarrollo ejecutados.
- GLM integrado al backtesting.
- Random Forest integrado.
- XGBoost integrado.
- Mismo target y monto utilizados para mantener comparabilidad.
- Predicciones y resultados estandarizados.
- Tests asociados superados.
- Notebook `04_backtesting.ipynb` ejecutado.
- Resultados revisados y comentarios finales incorporados.

## Próxima fase

### 05 — Evaluación y comparación

Objetivos:

- consolidar resultados de todos los modelos;
- definir métricas principales;
- comparar por escenario;
- comparar por edad de desarrollo;
- analizar bias y estabilidad;
- identificar fortalezas y limitaciones de cada enfoque;
- preparar la selección y las conclusiones del proyecto.

## Estado por fase

| Fase | Estado |
|---|---|
| 00 — Roadmap y estado | 🟢 Continuo |
| 01 — Diseño y simulación | ✅ Completada |
| 02 — Validación y exploración | ✅ Completada |
| 03 — Diseño del experimento | ✅ Completada |
| 04 — Backtesting | ✅ Completada |
| 05 — Evaluación y comparación | 🟡 Siguiente |
| 06 — Reserva y reporte final | ⚪ Pendiente |

## Próximo objetivo

Construir la comparación global de métodos a partir de los resultados generados en el backtesting.