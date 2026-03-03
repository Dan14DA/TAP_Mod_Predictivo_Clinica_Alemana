"""
Preparación de Datos para Power BI
Estructura Dimensional Optimizada
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("PREPARACIÓN DE DATOS PARA POWER BI")
print("Dashboard de Mantenimiento Predictivo - Clínica Alemana")
print("="*80)

# ============================================================================
# CARGAR DATOS
# ============================================================================

print("\n[1/7] Cargando datos...")

df_completo = pd.read_csv('/home/claude/datos_luminarias_clinica_alemana.csv',
                          parse_dates=['fecha_instalacion', 'fecha_falla', 'fecha_cambio'])

df_predicciones = pd.read_csv('/home/claude/predicciones_mantenimiento.csv')

print(f"   ✓ Datos completos: {len(df_completo):,} registros")
print(f"   ✓ Predicciones: {len(df_predicciones):,} registros")

# ============================================================================
# TABLA 1: FACT_LUMINARIAS (Tabla de Hechos Principal)
# ============================================================================

print("\n[2/7] Creando FACT_Luminarias...")

# Combinar datos completos con predicciones
fact_luminarias = df_completo.copy()

# Merge con predicciones (solo para operativas)
fact_luminarias = fact_luminarias.merge(
    df_predicciones[['id_luminaria', 'dias_restantes_predichos', 
                     'prob_falla_alto', 'prob_falla_medio', 'prob_falla_bajo']],
    on='id_luminaria',
    how='left'
)

# Seleccionar y renombrar columnas clave
fact_luminarias = fact_luminarias[[
    'id_luminaria',
    'fecha_instalacion',
    'horas_uso_acumuladas',
    'porcentaje_vida_consumida',
    'vida_util_real_calculada',
    'ciclos_acumulados',
    'temperatura_promedio_c',
    'humedad_promedio_pct',
    'voltaje_promedio_v',
    'dias_restantes_estimados',
    'dias_restantes_predichos',
    'horas_restantes_estimadas',
    'nivel_riesgo',
    'prioridad_mantenimiento',
    'estado_actual',
    'fecha_falla',
    'fecha_cambio',
    'motivo_cambio',
    'prob_falla_alto',
    'prob_falla_medio',
    'prob_falla_bajo'
]]

# Rellenar NaN en probabilidades (luminarias reemplazadas)
fact_luminarias['prob_falla_alto'] = fact_luminarias['prob_falla_alto'].fillna(0)
fact_luminarias['prob_falla_medio'] = fact_luminarias['prob_falla_medio'].fillna(0)
fact_luminarias['prob_falla_bajo'] = fact_luminarias['prob_falla_bajo'].fillna(0)

print(f"   ✓ FACT_Luminarias creada: {len(fact_luminarias):,} registros, {len(fact_luminarias.columns)} columnas")

# ============================================================================
# TABLA 2: DIM_UBICACION (Dimensión de Ubicación)
# ============================================================================

print("\n[3/7] Creando DIM_Ubicacion...")

dim_ubicacion = df_completo[['id_luminaria', 'piso', 'area', 
                              'ubicacion_especifica', 'criticidad']].copy()

# Crear ID único para ubicación
dim_ubicacion['id_ubicacion'] = range(1, len(dim_ubicacion) + 1)

# Ordenar columnas
dim_ubicacion = dim_ubicacion[[
    'id_ubicacion',
    'id_luminaria',
    'piso',
    'area',
    'ubicacion_especifica',
    'criticidad'
]]

print(f"   ✓ DIM_Ubicacion creada: {len(dim_ubicacion):,} registros")

# ============================================================================
# TABLA 3: DIM_PRODUCTO (Dimensión de Producto LED)
# ============================================================================

print("\n[4/7] Creando DIM_Producto...")

dim_producto = df_completo[['id_luminaria', 'tipo_led', 'marca', 
                            'potencia_watts', 'vida_util_nominal', 
                            'horas_operacion_dia']].copy()

# ID único para producto
dim_producto['id_producto'] = range(1, len(dim_producto) + 1)

dim_producto = dim_producto[[
    'id_producto',
    'id_luminaria',
    'tipo_led',
    'marca',
    'potencia_watts',
    'vida_util_nominal',
    'horas_operacion_dia'
]]

print(f"   ✓ DIM_Producto creada: {len(dim_producto):,} registros")

# ============================================================================
# TABLA 4: DIM_TIEMPO (Dimensión de Tiempo)
# ============================================================================

print("\n[5/7] Creando DIM_Tiempo...")

# Generar rango de fechas completo
fecha_min = df_completo['fecha_instalacion'].min()
fecha_max = pd.to_datetime('2026-12-31')  # Extender para proyecciones
fechas = pd.date_range(start=fecha_min, end=fecha_max, freq='D')

dim_tiempo = pd.DataFrame({
    'Fecha': fechas
})

dim_tiempo['Año'] = dim_tiempo['Fecha'].dt.year
dim_tiempo['Mes'] = dim_tiempo['Fecha'].dt.month
dim_tiempo['MesNombre'] = dim_tiempo['Fecha'].dt.strftime('%B')
dim_tiempo['Trimestre'] = dim_tiempo['Fecha'].dt.quarter
dim_tiempo['Semana'] = dim_tiempo['Fecha'].dt.isocalendar().week
dim_tiempo['DiaSemana'] = dim_tiempo['Fecha'].dt.dayofweek + 1
dim_tiempo['DiaSemanaNombre'] = dim_tiempo['Fecha'].dt.strftime('%A')
dim_tiempo['DiaDelAño'] = dim_tiempo['Fecha'].dt.dayofyear

print(f"   ✓ DIM_Tiempo creada: {len(dim_tiempo):,} registros (desde {fecha_min.date()} hasta {fecha_max.date()})")

# ============================================================================
# TABLA 5: FACT_MANTENIMIENTO_HISTORICO (Historial de Mantenimientos)
# ============================================================================

print("\n[6/7] Creando FACT_Mantenimiento_Historico...")

# Solo luminarias que han sido reemplazadas
mantenimientos = df_completo[df_completo['estado_actual'] == 'REEMPLAZADO'].copy()

fact_mantenimiento = mantenimientos[[
    'id_luminaria',
    'fecha_instalacion',
    'fecha_falla',
    'fecha_cambio',
    'motivo_cambio',
    'horas_uso_acumuladas',
    'vida_util_real_calculada'
]].copy()

# Calcular días hasta falla
fact_mantenimiento['dias_hasta_falla'] = (
    fact_mantenimiento['fecha_falla'] - fact_mantenimiento['fecha_instalacion']
).dt.days

# Calcular días de respuesta
fact_mantenimiento['dias_respuesta'] = (
    fact_mantenimiento['fecha_cambio'] - fact_mantenimiento['fecha_falla']
).dt.days

# ID de mantenimiento
fact_mantenimiento['id_mantenimiento'] = range(1, len(fact_mantenimiento) + 1)

fact_mantenimiento = fact_mantenimiento[[
    'id_mantenimiento',
    'id_luminaria',
    'fecha_instalacion',
    'fecha_falla',
    'fecha_cambio',
    'dias_hasta_falla',
    'dias_respuesta',
    'motivo_cambio',
    'horas_uso_acumuladas',
    'vida_util_real_calculada'
]]

print(f"   ✓ FACT_Mantenimiento_Historico creada: {len(fact_mantenimiento):,} registros")

# ============================================================================
# TABLA 6: RESUMEN_METRICAS (Tabla de Resumen para KPIs)
# ============================================================================

print("\n[7/7] Creando Resumen_Metricas...")

operativas = df_completo[df_completo['estado_actual'] == 'OPERATIVO']

resumen_metricas = pd.DataFrame({
    'Metrica': [
        'Total_Luminarias',
        'Luminarias_Operativas',
        'Luminarias_Reemplazadas',
        'Riesgo_MUY_ALTO',
        'Riesgo_ALTO',
        'Riesgo_MEDIO',
        'Riesgo_BAJO',
        'Prioridad_1_Urgente',
        'Prioridad_2_Alta',
        'Areas_CRITICAS',
        'Vida_Util_Promedio_Pct',
        'Horas_Uso_Promedio',
        'Tasa_Falla_Global_Pct'
    ],
    'Valor': [
        len(df_completo),
        len(operativas),
        len(df_completo[df_completo['estado_actual'] == 'REEMPLAZADO']),
        len(operativas[operativas['nivel_riesgo'] == 'MUY ALTO']),
        len(operativas[operativas['nivel_riesgo'] == 'ALTO']),
        len(operativas[operativas['nivel_riesgo'] == 'MEDIO']),
        len(operativas[operativas['nivel_riesgo'] == 'BAJO']),
        len(operativas[operativas['prioridad_mantenimiento'] == 1]),
        len(operativas[operativas['prioridad_mantenimiento'] == 2]),
        len(df_completo[df_completo['criticidad'] == 'CRÍTICA']),
        round(df_completo['porcentaje_vida_consumida'].mean(), 2),
        round(df_completo['horas_uso_acumuladas'].mean(), 0),
        round((len(df_completo[df_completo['estado_actual'] == 'REEMPLAZADO']) / len(df_completo)) * 100, 2)
    ]
})

print(f"   ✓ Resumen_Metricas creada: {len(resumen_metricas)} métricas")

# ============================================================================
# GUARDAR ARCHIVOS
# ============================================================================

print("\n" + "="*80)
print("GUARDANDO ARCHIVOS PARA POWER BI")
print("="*80)

# Guardar como Excel con múltiples hojas
excel_path = '/home/claude/PowerBI_Dashboard_Data.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    fact_luminarias.to_excel(writer, sheet_name='FACT_Luminarias', index=False)
    dim_ubicacion.to_excel(writer, sheet_name='DIM_Ubicacion', index=False)
    dim_producto.to_excel(writer, sheet_name='DIM_Producto', index=False)
    dim_tiempo.to_excel(writer, sheet_name='DIM_Tiempo', index=False)
    fact_mantenimiento.to_excel(writer, sheet_name='FACT_Mantenimiento', index=False)
    resumen_metricas.to_excel(writer, sheet_name='Resumen_Metricas', index=False)

print(f"\n✓ Archivo principal guardado: PowerBI_Dashboard_Data.xlsx")

# También guardar CSVs individuales
fact_luminarias.to_csv('/home/claude/PowerBI_FACT_Luminarias.csv', index=False, encoding='utf-8-sig')
dim_ubicacion.to_csv('/home/claude/PowerBI_DIM_Ubicacion.csv', index=False, encoding='utf-8-sig')
dim_producto.to_csv('/home/claude/PowerBI_DIM_Producto.csv', index=False, encoding='utf-8-sig')
dim_tiempo.to_csv('/home/claude/PowerBI_DIM_Tiempo.csv', index=False, encoding='utf-8-sig')
fact_mantenimiento.to_csv('/home/claude/PowerBI_FACT_Mantenimiento.csv', index=False, encoding='utf-8-sig')
resumen_metricas.to_csv('/home/claude/PowerBI_Resumen_Metricas.csv', index=False, encoding='utf-8-sig')

print(f"✓ CSVs individuales guardados (6 archivos)")

# ============================================================================
# RESUMEN DE ESTRUCTURA
# ============================================================================

print("\n" + "="*80)
print("ESTRUCTURA DE DATOS PARA POWER BI")
print("="*80)

print("\n📊 TABLAS CREADAS:")
print("\n1. FACT_Luminarias (Tabla de Hechos Principal)")
print(f"   • Registros: {len(fact_luminarias):,}")
print(f"   • Columnas: {', '.join(fact_luminarias.columns[:8])}...")
print(f"   • Clave: id_luminaria")

print("\n2. DIM_Ubicacion (Dimensión de Ubicación)")
print(f"   • Registros: {len(dim_ubicacion):,}")
print(f"   • Columnas: {', '.join(dim_ubicacion.columns)}")
print(f"   • Clave: id_luminaria")

print("\n3. DIM_Producto (Dimensión de Producto)")
print(f"   • Registros: {len(dim_producto):,}")
print(f"   • Columnas: {', '.join(dim_producto.columns)}")
print(f"   • Clave: id_luminaria")

print("\n4. DIM_Tiempo (Dimensión Calendario)")
print(f"   • Registros: {len(dim_tiempo):,}")
print(f"   • Rango: {dim_tiempo['Fecha'].min().date()} a {dim_tiempo['Fecha'].max().date()}")
print(f"   • Clave: Fecha")

print("\n5. FACT_Mantenimiento_Historico")
print(f"   • Registros: {len(fact_mantenimiento):,}")
print(f"   • Columnas: {', '.join(fact_mantenimiento.columns)}")
print(f"   • Clave: id_mantenimiento")

print("\n6. Resumen_Metricas (KPIs)")
print(f"   • Métricas: {len(resumen_metricas)}")

print("\n" + "="*80)
print("\n✅ DATOS PREPARADOS PARA POWER BI")
print("\nArchivos listos para importar:")
print("  • PowerBI_Dashboard_Data.xlsx (archivo principal con todas las tablas)")
print("  • 6 archivos CSV individuales")
print("\n" + "="*80)
