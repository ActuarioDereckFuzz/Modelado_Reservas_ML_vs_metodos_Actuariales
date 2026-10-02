# Modelado de Reservas: Machine Learning vs. Métodos Actuariales

Proyecto de modelación de **Loss Incurred** orientado a comparar métodos tradicionales de desarrollo, modelos estadísticos y técnicas de Machine Learning mediante un esquema de **backtesting temporal**.

El objetivo es estimar el Loss Incurred ultimate y la reserva pendiente bajo distintos patrones de desarrollo, manteniendo una separación estricta entre la información disponible en cada fecha de valuación y los resultados observados posteriormente.

---

## Objetivo

El proyecto estudia tres escenarios simulados de desarrollo mensual:

- **Creciente:** el Loss Incurred parte por debajo de su nivel definitivo y aumenta progresivamente hasta alcanzar madurez.
- **Decreciente:** parte por encima del nivel definitivo y presenta ajustes posteriores a la baja.
- **Mixto:** combina crecimiento temprano con ajustes posteriores decrecientes.

Para cada escenario se comparan tres familias de modelos:

1. **Métodos de desarrollo actuarial**
2. **Modelos Lineales Generalizados (GLM)**
3. **Modelos de Machine Learning**

La comparación se realiza mediante backtesting temporal, reproduciendo la información que habría estado disponible en distintas fechas históricas de valuación.

---

## Diseño de la simulación

Se generan bases mensuales reproducibles incorporando:

- tendencia;
- estacionalidad;
- variación aleatoria por período de ocurrencia;
- ruido durante el desarrollo;
- diferentes patrones de evolución del Loss Incurred.

Los horizontes de madurez utilizados son:

| Escenario | `dev_M` | Patrón |
|---|---:|---|
| Creciente | 60 meses | Desarrollo positivo |
| Decreciente | 48 meses | Ajustes negativos |
| Mixto | 48 meses | Crecimiento y ajuste posterior |

Cada período de ocurrencia cuenta con una trayectoria completa hasta `dev_M`, permitiendo conocer la verdad simulada y utilizarla posteriormente para evaluar las predicciones.

---

## Metodología

### 1. Simulación y validación

Se construyen las bases longitudinales de Loss Incurred y se valida:

- estructura temporal;
- períodos de ocurrencia y desarrollo;
- positividad de importes;
- madurez;
- comportamiento de los factores edad-a-edad.

Las bases se transforman posteriormente en:

- triángulos acumulados;
- triángulos incrementales;
- diagonales de valuación;
- tablas de factores de desarrollo.

### 2. Diseño temporal del experimento

El backtesting reproduce múltiples valuaciones históricas.

Para cada fecha se separan explícitamente:

- **training:** cohortes cuyo Loss Incurred a madurez ya sería conocido;
- **prediction:** cohortes todavía inmaduras;
- **target:** resultado a `dev_M`, revelado únicamente después de generar la predicción.

Esta arquitectura evita utilizar información futura durante el entrenamiento o la construcción de las predicciones.

### 3. Modelos de desarrollo

Se evalúan distintas especificaciones combinando:

- métodos de agregación de factores;
- ventanas históricas;
- reglas de exclusión.

A partir de los factores edad-a-edad se construyen CDFs para proyectar el Loss Incurred observado hasta su nivel ultimate.

### 4. GLM

Se incorporan modelos estadísticos como alternativa al enfoque tradicional de desarrollo, manteniendo las mismas cohortes y fechas de valuación utilizadas por las demás familias.

### 5. Machine Learning

Se evalúan modelos de Machine Learning dentro del mismo diseño experimental, utilizando exactamente la información disponible en cada snapshot histórico.

Esto permite comparar las distintas familias bajo condiciones temporales equivalentes.

### 6. Backtesting y selección

Las predicciones se evalúan mediante métricas de error a nivel de ultimate y reserva.

Entre las métricas analizadas se encuentran:

- MAE;
- RMSE;
- WAPE;
- sesgo;
- error agregado de reserva;
- estabilidad entre valuaciones.

La selección final se realiza utilizando exclusivamente los resultados históricos del backtesting.

---

## Resultados principales

La especificación de desarrollo **`VW_all_none`** fue seleccionada para los tres escenarios.

En el backtesting obtuvo los siguientes WAPE promedio:

| Escenario | Desarrollo | GLM | Machine Learning |
|---|---:|---:|---:|
| Creciente | **1.25%** | 5.45% | 5.27% |
| Decreciente | **1.21%** | 2.35% | 2.35% |
| Mixto | **1.22%** | 7.39% | 7.47% |

Estos resultados deben interpretarse dentro del diseño del experimento: las bases fueron simuladas mediante estructuras explícitas de desarrollo, por lo que los métodos tradicionales cuentan con una ventaja estructural y los resultados no implican superioridad general frente a GLM o Machine Learning en datos reales.

---

## Valuación final

Una vez seleccionado el método mediante backtesting, `VW_all_none` se aplica sobre la información disponible en la valuación final de **diciembre de 2025**.

La reserva se define como:

\[
\widehat{R}_i =
\widehat{U}_i - LI_i^{obs}
\]

donde:

- \(\widehat{U}_i\) es el Loss Incurred ultimate estimado;
- \(LI_i^{obs}\) es el Loss Incurred observado a la fecha de valuación.

Los resultados agregados fueron:

| Escenario | Reserva estimada | Reserva real* | Error |
|---|---:|---:|---:|
| Creciente | $11,241,642 | $11,005,479 | $236,163 |
| Decreciente | -$5,030,939 | -$5,309,903 | $278,964 |
| Mixto | -$133,970 | -$348,674 | $214,704 |

\* La reserva real corresponde a la verdad simulada y se revela únicamente después de congelar las predicciones. No participa en el entrenamiento, selección o estimación de la reserva.

### Interpretación del escenario mixto

La reserva neta cercana a cero no implica ausencia de desarrollo.

Las cohortes inmaduras presentan aproximadamente:

- **+$3.92 millones** de desarrollo positivo;
- **-$4.05 millones** de desarrollo negativo;
- **$7.97 millones** de movimiento bruto.

La reserva neta de aproximadamente -$0.13 millones surge de la compensación entre ambos componentes.

---

## Estructura del repositorio

```text
Modelado_Reservas_ML_vs_metodos_Actuariales/
│
├── data/
│   └── processed/
│
├── notebooks/
│   ├── 01_diseno_simulacion.ipynb
│   ├── 02_validacion_exploracion.ipynb
│   ├── 03_diseno_experimento.ipynb
│   ├── 04_backtesting.ipynb
│   ├── 05_evaluacion_seleccion.ipynb
│   └── 06_reserva_reporte_final.ipynb
│
├── src/
│   ├── analysis/
│   ├── experiment/
│   ├── models/
│   ├── simulation/
│   ├── triangles/
│   ├── validation/
│   └── visualization/
│
├── tests/
│
├── DECISIONS.md
├── PROJECT_STATE.md
├── README.md
└── ROADMAP.md
```

### Notebooks

| Notebook | Contenido |
|---|---|
| `01_diseno_simulacion.ipynb` | Diseño y generación de los escenarios simulados |
| `02_validacion_exploracion.ipynb` | Validación, triángulos y análisis exploratorio |
| `03_diseno_experimento.ipynb` | Diseño del backtesting y controles temporales |
| `04_backtesting.ipynb` | Ejecución de Desarrollo, GLM y Machine Learning |
| `05_evaluacion_seleccion.ipynb` | Comparación y selección de modelos |
| `06_reserva_reporte_final.ipynb` | Valuación final, diagnóstico y conclusiones |

La lógica reutilizable se mantiene en `src/`, mientras que los notebooks funcionan como capa de análisis, experimentación y presentación de resultados.

---

## Validación y pruebas

El proyecto incluye pruebas unitarias y de integración para componentes como:

- simulación;
- construcción de triángulos;
- factores de desarrollo;
- snapshots temporales;
- elegibilidad de cohortes;
- modelos;
- backtesting;
- integración del experimento.

Una parte central de las pruebas verifica la separación entre información observable y targets futuros para reducir el riesgo de **data leakage**.

---

## Flujo del proyecto

```text
Simulación
    ↓
Validación y exploración
    ↓
Triángulos de desarrollo
    ↓
Diseño temporal del experimento
    ↓
Development / GLM / Machine Learning
    ↓
Backtesting
    ↓
Evaluación y selección
    ↓
Valuación final
```

---

## Reproducibilidad

La simulación utiliza semillas pseudoaleatorias definidas en la configuración del proyecto, permitiendo reproducir las mismas bases y resultados.

El flujo recomendado de ejecución es:

```text
01_diseno_simulacion.ipynb
        ↓
02_validacion_exploracion.ipynb
        ↓
03_diseno_experimento.ipynb
        ↓
04_backtesting.ipynb
        ↓
05_evaluacion_seleccion.ipynb
        ↓
06_reserva_reporte_final.ipynb
```

La lógica principal utilizada por los notebooks se encuentra modularizada dentro de `src/`.

---

## Limitaciones

Los resultados deben interpretarse considerando las siguientes limitaciones:

- Los datos son **simulados** y siguen estructuras explícitas de desarrollo.
- Esta estructura favorece naturalmente a los métodos tradicionales de desarrollo.
- `dev_M` se considera conocido y fijo para cada escenario.
- Se supone estabilidad de los patrones históricos utilizados para proyectar cohortes inmaduras.
- Las edades más jóvenes de la valuación final no pueden evaluarse completamente mediante backtesting cuando su target todavía no se ha revelado dentro del horizonte histórico.
- La reserva estimada es determinista; no se construye una distribución probabilística de la incertidumbre de reserva.
- Una reserva neta cercana a cero puede ocultar movimientos positivos y negativos relevantes entre cohortes.

Por estas razones, el objetivo del proyecto no es demostrar que una familia de modelos sea universalmente superior, sino construir un marco reproducible para comparar metodologías bajo condiciones temporales consistentes.

---

## Tecnologías

- Python
- pandas
- NumPy
- Matplotlib
- statsmodels
- scikit-learn
- Jupyter
- pytest

---

## Conclusión

El proyecto integra conceptos actuariales de desarrollo de pérdidas con herramientas de modelación estadística y Machine Learning dentro de un experimento temporal reproducible.

Más allá de la comparación de métricas, el foco se encuentra en mantener una separación clara entre **entrenamiento, predicción, revelación del target y valuación final**, permitiendo evaluar las metodologías sin utilizar información futura y conservando una interpretación actuarial de los resultados.