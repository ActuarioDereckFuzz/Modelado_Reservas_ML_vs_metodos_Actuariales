# Project State — Loss Incurred

**Última actualización:** 2026-10-01

## Estado general

🟡 **En desarrollo**

## Fase actual

**01 — Diseño y simulación**

Se está revisando el diseño del proyecto antes de continuar con la implementación definitiva.

## Completado

- Definición de tres escenarios: creciente, decreciente y mixto.
- Primera versión de las bases simuladas.
- Primera implementación de métodos de desarrollo.
- GLM Gamma con enlace log.
- Primera versión del backtesting.
- Validación de claves comunes entre modelos.

## Problema identificado

El diseño inicial del backtesting excluye edades jóvenes debido a que el target debe estar completamente revelado antes de `2025-12`.

Esto puede sesgar la evaluación de los modelos.

## Prioridad actual

Revisar:

1. período de simulación;
2. `dev_M` de cada escenario;
3. definición del target;
4. diseño temporal del backtesting.

## Siguiente paso

Cerrar **Diseño y simulación** y posteriormente avanzar a:

**02 — Exploración y construcción de triángulos**.