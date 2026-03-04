# 💡 Sistema de Mantenimiento Predictivo de Iluminación LED
## Clínica Alemana de Santiago

![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Python](https://img.shields.io/badge/Python%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Excel](https://img.shields.io/badge/Excel-217346?style=for-the-badge&logo=microsoft-excel&logoColor=white)

> **Trabajo de Aplicación Práctica (TAP) — AIEP Data Science**  
> Autor: Danny Alberto Contreras Pérez | Carrera: Data Science | Sede: TLP-Online | Santiago, Febrero 2026

---

## 📋 Descripción del Proyecto

Sistema integral de **mantenimiento predictivo** para la infraestructura de iluminación LED de la Clínica Alemana de Santiago. Integra técnicas de **machine learning**, análisis de supervivencia y visualización avanzada para optimizar operaciones de mantenimiento, reducir costos y minimizar riesgos en áreas críticas (quirófanos, UCI, urgencias).

El sistema analiza **3,395 luminarias LED** distribuidas en 16 pisos, procesando variables operacionales, ambientales y de rendimiento para generar predicciones precisas y recomendaciones accionables, permitiendo la transición de un modelo **reactivo → proactivo**.

---

## 🎯 Resultados Principales

| Métrica | Resultado |
|--------|-----------|
| **R² Modelo Regresión** (Gradient Boosting) | **0.9944** — Error promedio 36.45 días |
| **Accuracy Clasificación de Riesgo** (Random Forest) | **99.59%** |
| **C-Index Análisis Supervivencia** (Cox PH) | **0.8282** |
| Luminarias en riesgo crítico identificadas | **294** (11.9% de operativas) |
| Intervención urgente (Prioridad 1) | **239 luminarias** |
| Proyección de compras a 1 año | **422 focos / $5.768.780 CLP** |
| Potencial reducción de costos | **60%** (preventivo vs reactivo) |
| ROI proyectado a 12 meses | **250–300%** |
| Ahorro anual estimado | **$45–60 millones CLP** |

---

## 🏗️ Arquitectura del Sistema

### Modelo Dimensional (Esquema Estrella)

```
              DIM_Tiempo
                  │
DIM_Ubicacion ── FACT_Luminarias ── DIM_Producto
                  │
            FACT_Mantenimiento
                  │
           Resumen_Metricas
```

| Tabla | Registros | Descripción |
|-------|-----------|-------------|
| `FACT_Luminarias` | 3,395 | Estado actual, predicciones y variables operacionales |
| `DIM_Ubicacion` | 3,395 | Piso, área y criticidad (CRÍTICA/ALTA/MEDIA/BAJA) |
| `DIM_Producto` | 3,395 | Tipo LED, marca, especificaciones técnicas |
| `DIM_Tiempo` | 2,708 | Calendario 2019–2026 |
| `FACT_Mantenimiento` | 933 | Historial de fallas y reemplazos |
| `Resumen_Metricas` | — | KPIs agregados del sistema |

---

## 📊 Dashboard Power BI

3 páginas de análisis interactivo con **29 medidas DAX** y **5 relaciones** en modelo estrella:

### Página 1 — Resumen Ejecutivo
- 6 KPI Cards: Total luminarias, Operatividad %, Riesgo Crítico, Prioridad 1, Indicador Salud
- Distribución por nivel de riesgo (BAJO / MEDIO / ALTO / MUY ALTO)
- Criticidad de áreas
- TreeMap por criticidad y Piso
- Top 10 áreas con mayor riesgo

### Página 2 — Análisis Predictivo
- Predicciones de vida útil remanente por luminaria
- Priorización de intervenciones
- Probabilidad de falla por área y tipo LED

### Página 3 — Mantenimiento Histórico
- Tendencias temporales de fallas
- Desempeño por marca y tipo de LED
- Tasa de falla global (27.5%)

---

## 🤖 Modelos de Machine Learning

### 1. Modelo de Regresión — Días Restantes hasta Falla

| Modelo | R² | MAE (días) | RMSE (días) |
|--------|-----|-----------|------------|
| Regresión Lineal | 0.9461 | 196.22 | 253.67 |
| Random Forest | 0.9855 | 52.15 | 131.39 |
| **Gradient Boosting** ✅ | **0.9944** | **36.45** | **81.83** |

**Variables más importantes:**
1. % Vida consumida: **86.86%**
2. Horas operación diaria: 7.92%
3. Vida útil real calculada: 3.74%
4. Ciclos acumulados: 1.15%
5. Criticidad del área: 0.10%

### 2. Modelo de Clasificación de Riesgo
- **Algoritmo:** Random Forest Classifier
- **Accuracy:** 99.59%
- **Clases:** ALTO / MEDIO / BAJO
- Validación cruzada 5-folds, train/test 80/20

### 3. Análisis de Supervivencia
- **Kaplan-Meier:** Curvas por tipo LED
- **Cox Proportional Hazards:** C-Index = 0.8282
- LED Quirúrgico: mejor desempeño (3.5% tasa de falla)
- LED Tubo T8: mayor tasa de falla (52.9%)

---

## 📦 Planificación Operativa

### Sistema de Proyección de Compras

| Período | Focos | Inversión (CLP) |
|---------|-------|----------------|
| 30 días | 57 | $703,430 |
| 60 días | 74 | $874,260 |
| 90 días | 92 | $1,109,080 |
| 180 días | 183 | $2,143,170 |
| **1 año** | **422** | **$5,768,780** |

**Desglose por tipo (próximo año):**
- LED Downlight: 200 u. ($1,998,000)
- LED Tubo T8: 118 u. ($1,060,820)
- LED Emergencia: 50 u. ($799,500)
- LED Industrial: 28 u. ($531,720)
- LED Panel 60×60: 18 u. ($233,820)
- LED Quirúrgico: 8 u. ($367,920)

### Sistema de Asignación de Trabajadores
- **10 trabajadores eléctricos**, distribución equitativa con algoritmo Greedy
- 92 luminarias asignadas (próximos 90 días)
- **2.48 horas promedio** por trabajador | **CV = 3.0%** (excelente balance)
- Agrupación por sectores: A (pisos bajos), B (medios), C (altos), D (muy altos)

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Tecnologías |
|-----------|------------|
| **Lenguaje** | Python 3.12 |
| **ML / Estadística** | scikit-learn, lifelines, numpy |
| **Análisis de Datos** | pandas, scipy |
| **Visualización Python** | matplotlib, seaborn, plotly |
| **Dashboard** | Microsoft Power BI Desktop (DAX, AMO/XMLA) |
| **Almacenamiento** | Excel 365, CSV |
| **Automatización PBI** | PowerShell + AMO (Analysis Management Objects) |

---

## 📁 Estructura del Repositorio

```
TAP/
├── 📄 Clinica Alemana.pbix          # Dashboard Power BI (modelo + medidas DAX)
├── 📁 Codigo_Python/                 # Scripts Python del proyecto
│   ├── 01_generacion_datos_clinica_alemana.py
│   ├── 02_analisis_exploratorio.py
│   ├── 03_modelos_predictivos.py
│   ├── 04_preparar_datos_powerbi.py
│   └── 05_planificacion_operativa.py
├── 📁 File/                          # Datos y dashboards HTML de referencia
│   ├── CSV/                          # Tablas del modelo dimensional
│   │   ├── PowerBI_FACT_Luminarias.csv
│   │   ├── PowerBI_FACT_Mantenimiento.csv
│   │   ├── PowerBI_DIM_Ubicacion.csv
│   │   ├── PowerBI_DIM_Producto.csv
│   │   ├── PowerBI_DIM_Tiempo.csv
│   │   └── PowerBI_Resumen_Metricas.csv
│   ├── PowerBI_Dashboard_Data.xlsx   # Fuente de datos para Power BI
│   ├── dashboard_1_resumen_ejecutivo
│   ├── dashboard_2_analisis_predictivo.html
│   └── dashboard_3_mantenimineto historico.html
├── 📁 Instrucciones/                 # Documentación de uso
└── 📁 Reporte_Final/                 # Informe académico TAP
    ├── Informe_TAP_AIEP_DATA_SCIENCE.pdf
    ├── Guia Técnica
    └── Guia de Usuario
```

---

## 🚀 Instrucciones de Uso

### Requisitos
- Python 3.12+ con: `pip install pandas numpy scikit-learn lifelines matplotlib seaborn plotly`
- Power BI Desktop (versión actual)
- Excel 365 (para visualizar los CSV)

### Paso 1 — Ejecutar Scripts Python
```bash
# Generar dataset simulado
python Codigo_Python/01_generacion_datos_clinica_alemana.py

# Análisis exploratorio
python Codigo_Python/02_analisis_exploratorio.py

# Entrenar modelos predictivos
python Codigo_Python/03_modelos_predictivos.py

# Preparar datos para Power BI
python Codigo_Python/04_preparar_datos_powerbi.py

# Planificación operativa
python Codigo_Python/05_planificacion_operativa.py
```

### Paso 2 — Abrir Power BI
1. Abrir `Clinica Alemana.pbix` en Power BI Desktop
2. El modelo ya incluye **5 relaciones** y **29 medidas DAX** precargadas


## 📈 Hallazgos Clave del EDA

- **72.5%** de luminarias en estado OPERATIVO (2,462 de 3,395)
- **77.2%** vida útil promedio consumida en luminarias activas
- **27.5%** tasa de falla global (933 reemplazadas)
- **Philips y Osram:** mejor desempeño (22–24% falla vs 32–37% marcas genéricas)
- **LED Quirúrgico:** especialidad más confiable (3.5% tasa de falla)
- **LED Tubo T8:** mayor tasa de falla (52.9%) → prioridad de evaluación

---

## Transición a Datos Reales

- Cuando esté listo para conectar datos reales de la Clínica Alemana:
- Exportar datos del sistema actual al formato Excel definido
- Ejecutar modelos predictivos en Python con datos reales
- Generar archivo PowerBI_Dashboard_Data.xlsx actualizado
- Actualizar conexión en Power BI
- Configurar actualización automática (diaria, semanal, mensual según necesidad)

---

## Mejoras Futuras Recomendadas
-	Alertas automáticas: Notificaciones cuando luminarias entren en riesgo MUY ALTO
-	Integración con sistema de tickets: Generar órdenes de trabajo automáticamente
-	Optimización de rutas: Algoritmo para planificar rutas eficientes de mantenimiento
-	Análisis de costos: Comparación entre mantenimiento preventivo vs reactivo
-	Dashboard móvil: Versión optimizada para tablets del equipo de mantenimiento

---

## 🏥 Contexto Institucional

**Clínica Alemana de Santiago** — Una de las instituciones de salud más importantes de Chile:
- +202,000 m² construidos (Torre principal Vitacura + Manquehue Oriente)
- 400 camas hospitalización | 22 pabellones quirúrgicos
- UCI, Urgencias general y escolar, Diagnóstico e imagenología
- Mantenimiento eléctrico a cargo de **Mantenciones Eléctricas H&C SpA**

---

## 📚 Referencias Normativas

- **IEC 62717:2014** — LED modules for general lighting
- **DS 594** — Condiciones Sanitarias y Ambientales en Lugares de Trabajo (Chile)
- **NCh Elec. 4/2003** — Instalaciones eléctricas hospitalarias

---

## 📄 Licencia

Proyecto académico desarrollado para Instituto Profesional AIEP — Carrera Data Science.  
© 2026 Danny Alberto Contreras Pérez

---

*Sistema desarrollado bajo un modelo de 4 fases en 6 meses · 480 horas de desarrollo · TAP AIEP 2026*


