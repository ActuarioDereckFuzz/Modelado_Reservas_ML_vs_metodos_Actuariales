# Project State — Loss Incurred

**Última actualización:** 2026-10-02

## Estado general

🟢 **Proyecto completado**

El desarrollo técnico, la valuación final y la documentación principal se
encuentran terminados.

## Fase actual

**Proyecto finalizado — mejoras futuras opcionales**

## Fases completadas

### 01 — Diseño y simulación ✅
- Tres escenarios: creciente, decreciente y mixto.
- Simulación mensual reproducible.
- `dev_M = 60 / 48 / 48`.
- Tendencia, estacionalidad, ruido y madurez validados.

### 02 — Validación y exploración ✅
- Estructura de las bases validada.
- Triángulos acumulados e incrementales.
- Diagonal más reciente.
- Factores edad-a-edad.
- Tendencia y estacionalidad analizadas.
- Patrones esperados corroborados.

### 03 — Diseño del experimento ✅
- Protocolo temporal definido.
- Snapshots históricos.
- Separación training / prediction / target.
- Controles contra data leakage.
- Tests unitarios e integración superados.

### 04 — Backtesting ✅
- Métodos de desarrollo evaluados.
- GLM evaluados.
- Modelos de Machine Learning evaluados.
- Predicciones estandarizadas.
- Comparabilidad entre familias garantizada.
- Backtesting completo para los tres escenarios.

### 05 — Evaluación y selección ✅
- Métricas consolidadas.
- Comparación por escenario.
- Familias de modelos comparadas.
- `VW_all_none` seleccionado para los tres escenarios.
- Gráficos y conclusiones incorporados.
- Notebook finalizado.

### 06 — Reserva y reporte final ✅
- Método seleccionado aplicado a la valuación final `2025-12`.
- Ultimate estimado para las cohortes inmaduras.
- Reserva estimada por cohorte y escenario.
- Composición positiva y negativa de la reserva analizada.
- Verdad simulada revelada únicamente después de congelar las predicciones.
- Error de reserva, MAE y RMSE evaluados.
- Notebook final limpiado, homogeneizado y ejecutado correctamente.

### 07 — Documentación y cierre ✅
- `README.md` actualizado para presentación del proyecto.
- Estructura del repositorio documentada.
- Metodología y resultados principales resumidos.
- Limitaciones documentadas.
- `ROADMAP.md` actualizado.
- `PROJECT_STATE.md` actualizado.
- Estructura final del repositorio revisada.

## Resultados finales

### Selección mediante backtesting

El método seleccionado para los tres escenarios fue:

`VW_all_none`

WAPE promedio observado en backtesting:

| Escenario | Desarrollo | GLM | Machine Learning |
|---|---:|---:|---:|
| Creciente | 1.25% | 5.45% | 5.27% |
| Decreciente | 1.21% | 2.35% | 2.35% |
| Mixto | 1.22% | 7.39% | 7.47% |

### Valuación final

| Escenario | Reserva estimada | Reserva real | Error |
|---|---:|---:|---:|
| Creciente | $11,241,642 | $11,005,479 | $236,163 |
| Decreciente | -$5,030,939 | -$5,309,903 | $278,964 |
| Mixto | -$133,970 | -$348,674 | $214,704 |

En el escenario mixto, la reserva neta cercana a cero surge de la compensación
entre aproximadamente $3.92 millones de desarrollo positivo y -$4.05 millones
de desarrollo negativo, equivalentes a aproximadamente $7.97 millones de
movimiento bruto.

## Estado por fase

| Fase | Estado |
|---|---|
| 00 — Roadmap y estado | ✅ Cerrada |
| 01 — Diseño y simulación | ✅ Completada |
| 02 — Validación y exploración | ✅ Completada |
| 03 — Diseño del experimento | ✅ Completada |
| 04 — Backtesting | ✅ Completada |
| 05 — Evaluación y selección | ✅ Completada |
| 06 — Reserva y reporte final | ✅ Completada |
| 07 — Documentación y cierre | ✅ Completada |

## Mejoras futuras

El proyecto se considera terminado dentro de su alcance actual.

Como mejoras posteriores pueden realizarse:

- crear un `requirements.txt` específico para el repositorio;
- comprobar reproducibilidad desde un entorno limpio;
- realizar ajustes menores de presentación para GitHub;
- extender el experimento a datos reales;
- incorporar cuantificación probabilística de la incertidumbre de reserva.

Estas mejoras no modifican los resultados ni son necesarias para considerar
completado el desarrollo actual.