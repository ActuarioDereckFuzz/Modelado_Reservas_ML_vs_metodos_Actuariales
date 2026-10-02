# Roadmap — Modelación de Loss Incurred

## Objetivo

Comparar métodos actuariales de desarrollo, modelos estadísticos y técnicas de
Machine Learning para proyectar el Loss Incurred mensual y estimar reservas bajo
tres escenarios simulados:

- creciente;
- decreciente;
- mixto.

La comparación se realiza mediante backtesting temporal, manteniendo una
separación explícita entre información disponible, predicción y revelación
posterior del target.

## Roadmap

| Fase | Descripción | Estado |
|---|---|---|
| 01 | Diseño y simulación | ✅ Completada |
| 02 | Validación y exploración | ✅ Completada |
| 03 | Diseño del experimento | ✅ Completada |
| 04 | Backtesting | ✅ Completada |
| 05 | Evaluación y selección | ✅ Completada |
| 06 | Reserva y reporte final | ✅ Completada |
| 07 | Documentación y cierre | ✅ Completada |

## Flujo

`Simulación → Validación → Triángulos → Diseño temporal → Modelos → Backtesting → Evaluación → Reserva final`

## Estado actual

El desarrollo técnico y la documentación principal del proyecto se encuentran
finalizados.

Los notebooks 01–06 contienen el flujo completo desde la generación de los
escenarios hasta la estimación y evaluación de la reserva final.

El repositorio cuenta además con código modular en `src/`, pruebas unitarias y
de integración en `tests/`, y documentación general mediante `README.md`,
`PROJECT_STATE.md`, `ROADMAP.md` y `DECISIONS.md`.

## Mejoras futuras

Las siguientes actividades se consideran mejoras opcionales y no forman parte
del alcance necesario para considerar terminado el proyecto:

- construir un `requirements.txt` específico a partir de las dependencias
  realmente utilizadas;
- validar la instalación desde un entorno limpio;
- realizar ajustes menores de presentación antes de publicar en GitHub.