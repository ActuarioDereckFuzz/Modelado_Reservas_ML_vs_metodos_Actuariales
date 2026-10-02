# Project State — Loss Incurred

**Última actualización:** 2026-10-02

## Estado general

🟢 **En desarrollo — Evaluación y selección completadas**

## Fase actual

**06 — Reserva y cierre del proyecto**

## Completado

### 01 — Diseño y simulación ✅
- Tres escenarios simulados.
- Periodicidad mensual.
- `dev_M = 60 / 48 / 48`.
- Tendencia, estacionalidad, ruido y madurez validados.
- Simulación reproducible y testeada.

### 02 — Validación y exploración ✅
- Bases y estructura temporal validadas.
- Triángulos acumulados e incrementales.
- Diagonal más reciente.
- Factores edad-a-edad.
- Tendencia y estacionalidad.
- Patrones de desarrollo corroborados.

### 03 — Diseño del experimento ✅
- Protocolo temporal definido.
- Snapshots históricos.
- Training / prediction / target separados.
- Controles de leakage.
- Elegibilidad y observabilidad correctamente diferenciadas.
- Tests unitarios e integración superados.

### 04 — Backtesting ✅
- Métodos de desarrollo evaluados.
- GLM integrado.
- Modelos de Machine Learning integrados.
- Predicciones estandarizadas.
- Mismas claves y targets para comparación.
- Backtesting ejecutado para los tres escenarios.
- Notebook final revisado y documentado.

### 05 — Evaluación y selección ✅
- Resultados del backtesting consolidados.
- Comparación entre familias de modelos.
- WAPE y métricas complementarias analizadas.
- Comparación gráfica incorporada.
- Selección de método por escenario.
- Conclusiones finales documentadas.
- Notebook `05_evaluacion_seleccion.ipynb` reorganizado y finalizado.

## Resultado principal de selección

El método de desarrollo `VW_all_none` presentó el menor WAPE medio en los tres escenarios:

| Escenario | Método seleccionado | WAPE medio |
|---|---|---:|
| Creciente | VW_all_none | 1.25% |
| Decreciente | VW_all_none | 1.21% |
| Mixto | VW_all_none | 1.22% |

Los modelos GLM y ML permanecen como benchmarks importantes para analizar el comportamiento relativo de enfoques más flexibles frente al método actuarial.

## Próxima fase

### 06 — Reserva y cierre

Objetivos principales:

- aplicar los métodos seleccionados a la valuación final;
- obtener las estimaciones finales;
- presentar resultados actuariales;
- consolidar conclusiones y limitaciones;
- limpiar notebooks y estructura del repositorio;
- completar `README.md`;
- preparar el proyecto para GitHub / presentación final.

## Estado por fase

| Fase | Estado |
|---|---|
| 00 — Roadmap y estado | 🟢 Continuo |
| 01 — Diseño y simulación | ✅ Completada |
| 02 — Validación y exploración | ✅ Completada |
| 03 — Diseño del experimento | ✅ Completada |
| 04 — Backtesting | ✅ Completada |
| 05 — Evaluación y selección | ✅ Completada |
| 06 — Reserva y cierre | 🟡 Siguiente |

## Próximo objetivo

Aplicar la metodología seleccionada a la valuación final y cerrar los entregables del proyecto.