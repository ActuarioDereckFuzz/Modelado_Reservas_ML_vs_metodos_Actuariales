# Project State — Loss Incurred

**Última actualización:** 2026-10-01

## Estado general

🟢 **En desarrollo — Fase 01 completada**

## Fase actual

**02 — Validación y estructura**

## Completado

### Fase 01 — Diseño y simulación ✅

- Tres escenarios simulados:
  - creciente;
  - decreciente;
  - mixto.
- Período de simulación:
  - `2015-01` a `2025-12`.
- Horizonte de madurez:
  - creciente: `dev_M = 60`;
  - decreciente: `dev_M = 48`;
  - mixto: `dev_M = 48`.
- Tendencia, estacionalidad y ruido incorporados.
- Reproducibilidad validada.
- Importes positivos validados.
- Calendarios y dimensiones validados.
- Madurez validada:
  - `LI_M = Ultimate`.
- Notebook `01_diseno_simulacion.ipynb` ejecutado correctamente.
- Módulo `src/simulation` validado.
- `30 tests` ejecutados correctamente.

## Decisiones vigentes

- Mantener `dev_M = 60 / 48 / 48`.
- No modificar la lógica de reproducibilidad incremental.
- Trabajar con datos mensuales.
- Mantener los tres patrones de desarrollo definidos.

## Próxima fase

### 02 — Validación y estructura

Objetivos principales:

- validar estructura de las bases;
- revisar períodos de ocurrencia y desarrollo;
- analizar factores edad-a-edad observados;
- comprobar que cada escenario reproduce el comportamiento esperado;
- preparar las bases para la construcción de triángulos.

## Estado por fase

| Fase | Estado |
|---|---|
| 00 — Roadmap y estado | 🟢 En progreso |
| 01 — Diseño y simulación | ✅ Completada |
| 02 — Validación y estructura | 🟡 Siguiente |
| 03 — Triángulos y métodos actuariales | ⚪ Pendiente |
| 04 — Modelación estadística / ML | ⚪ Pendiente |
| 05 — Backtesting | ⚪ Pendiente |
| 06 — Comparación y resultados | ⚪ Pendiente |
| 07 — Documentación final | ⚪ Pendiente |

## Próximo objetivo

Completar la validación estructural de los tres escenarios antes de avanzar a la construcción y comparación de métodos de desarrollo.