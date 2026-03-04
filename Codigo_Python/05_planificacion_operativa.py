"""
Planificación Operativa - Sistema de Mantenimiento Predictivo
Funcionalidades:
1. Proyección de compras de focos por período
2. Asignación óptima de carga de trabajo entre 10 trabajadores
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PLANIFICACIÓN OPERATIVA - MANTENIMIENTO PREDICTIVO")
print("Clínica Alemana de Santiago")
print("="*80)

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================

print("\n[1/5] Cargando datos de predicciones...")

# Cargar predicciones del modelo
df_predicciones = pd.read_csv('/home/predicciones_mantenimiento.csv')
print(f"   ✓ Datos cargados: {len(df_predicciones):,} luminarias operativas")

# Cargar datos completos para información adicional
df_completo = pd.read_csv('/home/datos_luminarias_clinica_alemana.csv',
                          parse_dates=['fecha_instalacion'])

# Merge para tener toda la información
df = df_predicciones.merge(
    df_completo[['id_luminaria', 'piso', 'area', 'ubicacion_especifica', 'horas_operacion_dia', 'criticidad']],
    on='id_luminaria',
    how='left',
    suffixes=('', '_completo')
)

# Resolver columnas duplicadas
if 'area_completo' in df.columns:
    df['area'] = df['area'].fillna(df['area_completo'])
    df = df.drop('area_completo', axis=1)
if 'criticidad_completo' in df.columns:
    df['criticidad'] = df['criticidad'].fillna(df['criticidad_completo'])
    df = df.drop('criticidad_completo', axis=1)

print(f"   ✓ Información completa disponible")

# ============================================================================
# 2. PROYECCIÓN DE COMPRAS DE FOCOS
# ============================================================================

print("\n[2/5] Generando proyección de compras...")

# Definir períodos de análisis
periodos = [
    {'dias': 30, 'nombre': '30 días (1 mes)'},
    {'dias': 60, 'nombre': '60 días (2 meses)'},
    {'dias': 90, 'nombre': '90 días (3 meses)'},
    {'dias': 180, 'nombre': '180 días (6 meses)'},
    {'dias': 365, 'nombre': '365 días (1 año)'}
]

# Calcular focos a reemplazar por período
proyeccion_compras = []

for periodo in periodos:
    dias = periodo['dias']
    nombre = periodo['nombre']
    
    # Filtrar luminarias que fallarán en este período
    focos_periodo = df[df['dias_restantes_predichos'] <= dias]
    
    # Agrupar por tipo de LED
    compras_por_tipo = focos_periodo.groupby('tipo_led').agg({
        'id_luminaria': 'count'
    }).reset_index()
    compras_por_tipo.columns = ['tipo_led', 'cantidad']
    compras_por_tipo['periodo'] = nombre
    compras_por_tipo['dias'] = dias
    
    proyeccion_compras.append(compras_por_tipo)
    
    # Resumen por período
    total = len(focos_periodo)
    print(f"\n   {nombre}:")
    print(f"      Total a reemplazar: {total:,} focos")
    
    # Mostrar top 3 tipos más demandados
    if len(compras_por_tipo) > 0:
        top3 = compras_por_tipo.nlargest(3, 'cantidad')
        for idx, row in top3.iterrows():
            print(f"      • {row['tipo_led']}: {row['cantidad']:.0f} unidades")

# Consolidar en DataFrame
df_proyeccion_compras = pd.concat(proyeccion_compras, ignore_index=True)

# Agregar información de criticidad
focos_criticos_por_periodo = []
for periodo in periodos:
    dias = periodo['dias']
    nombre = periodo['nombre']
    
    focos_periodo = df[df['dias_restantes_predichos'] <= dias]
    
    criticos = focos_periodo.groupby('criticidad').agg({
        'id_luminaria': 'count'
    }).reset_index()
    criticos.columns = ['criticidad', 'cantidad']
    criticos['periodo'] = nombre
    criticos['dias'] = dias
    
    focos_criticos_por_periodo.append(criticos)

df_criticos_periodo = pd.concat(focos_criticos_por_periodo, ignore_index=True)

# Calcular costos estimados (basado en tabla de costos)
costos_unitarios = {
    'LED Panel 60x60': 12990,
    'LED Tubo T8': 8990,
    'LED Downlight': 9990,
    'LED Quirúrgico': 45990,
    'LED Emergencia': 15990,
    'LED Industrial': 18990
}

df_proyeccion_compras['costo_unitario_clp'] = df_proyeccion_compras['tipo_led'].map(costos_unitarios)
df_proyeccion_compras['costo_total_clp'] = df_proyeccion_compras['cantidad'] * df_proyeccion_compras['costo_unitario_clp']

# Resumen de inversión por período
print("\n   " + "="*60)
print("   INVERSIÓN ESTIMADA EN FOCOS:")
print("   " + "="*60)
for periodo in periodos:
    nombre = periodo['nombre']
    inversion = df_proyeccion_compras[df_proyeccion_compras['periodo'] == nombre]['costo_total_clp'].sum()
    print(f"   {nombre}: ${inversion:,.0f} CLP")

# ============================================================================
# 3. CÁLCULO DE HORAS DE TRABAJO NECESARIAS
# ============================================================================

print("\n[3/5] Calculando carga de trabajo...")

# Tiempo promedio de reemplazo por tipo de LED (en minutos)
tiempo_reemplazo = {
    'LED Panel 60x60': 15,
    'LED Tubo T8': 12,
    'LED Downlight': 15,
    'LED Quirúrgico': 45,
    'LED Emergencia': 20,
    'LED Industrial': 25
}

# Agregar tiempo estimado a cada luminaria
df['tiempo_reemplazo_min'] = df['tipo_led'].map(tiempo_reemplazo)

# Calcular carga de trabajo por período
for periodo in periodos:
    dias = periodo['dias']
    nombre = periodo['nombre']
    
    focos_periodo = df[df['dias_restantes_predichos'] <= dias]
    
    horas_totales = (focos_periodo['tiempo_reemplazo_min'].sum()) / 60
    dias_trabajo = horas_totales / 8  # Jornada de 8 horas
    
    print(f"\n   {nombre}:")
    print(f"      Horas totales de trabajo: {horas_totales:,.1f} hrs")
    print(f"      Días-persona de trabajo: {dias_trabajo:,.1f} días")

# ============================================================================
# 4. ASIGNACIÓN DE TRABAJO A 10 TRABAJADORES
# ============================================================================

print("\n[4/5] Asignando trabajo a 10 trabajadores eléctricos...")

# Parámetros
NUM_TRABAJADORES = 10
PERIODO_ASIGNACION = 90  # días (3 meses)

# Filtrar luminarias para el período de asignación
df_asignar = df[df['dias_restantes_predichos'] <= PERIODO_ASIGNACION].copy()
df_asignar = df_asignar.sort_values('prioridad_mantenimiento')

print(f"\n   Total luminarias a asignar: {len(df_asignar):,}")
print(f"   Período: Próximos {PERIODO_ASIGNACION} días")

# Crear campo de sector (para agrupar por proximidad)
def asignar_sector(piso):
    if 'Piso -' in piso or piso in ['Piso 1', 'Piso 2', 'Piso 3']:
        return 'SECTOR_A_Bajo'
    elif piso in ['Piso 4', 'Piso 5', 'Piso 6', 'Piso 7']:
        return 'SECTOR_B_Medio'
    elif piso in ['Piso 8', 'Piso 9', 'Piso 10', 'Piso 11', 'Piso 12']:
        return 'SECTOR_C_Alto'
    else:
        return 'SECTOR_D_MuyAlto'

df_asignar['sector'] = df_asignar['piso'].apply(asignar_sector)

# Estrategia de asignación:
# 1. Distribuir equitativamente por carga de trabajo
# 2. Agrupar por sector para minimizar desplazamientos
# 3. Priorizar luminarias urgentes

# Calcular carga total
carga_total_minutos = df_asignar['tiempo_reemplazo_min'].sum()
carga_por_trabajador = carga_total_minutos / NUM_TRABAJADORES

print(f"\n   Carga total: {carga_total_minutos:,.0f} minutos ({carga_total_minutos/60:.1f} horas)")
print(f"   Carga objetivo por trabajador: {carga_por_trabajador:,.0f} minutos ({carga_por_trabajador/60:.1f} horas)")

# Asignación balanceada
trabajadores = []
for i in range(1, NUM_TRABAJADORES + 1):
    trabajadores.append({
        'id_trabajador': i,
        'nombre': f'Trabajador {i}',
        'carga_actual': 0,
        'luminarias': []
    })

# Ordenar por prioridad, criticidad y sector
df_asignar['score_asignacion'] = (
    df_asignar['prioridad_mantenimiento'] * 100 +  # Prioridad es lo más importante
    df_asignar['criticidad'].map({'CRÍTICA': 1, 'ALTA': 2, 'MEDIA': 3, 'BAJA': 4}) * 10
)
df_asignar = df_asignar.sort_values('score_asignacion')

# Asignar luminarias a trabajadores (algoritmo greedy)
asignaciones = []

for idx, luminaria in df_asignar.iterrows():
    # Encontrar trabajador con menor carga
    trabajador_min = min(trabajadores, key=lambda x: x['carga_actual'])
    
    # Asignar luminaria
    trabajador_min['carga_actual'] += luminaria['tiempo_reemplazo_min']
    trabajador_min['luminarias'].append(luminaria['id_luminaria'])
    
    # Registrar asignación
    asignaciones.append({
        'id_luminaria': luminaria['id_luminaria'],
        'id_trabajador': trabajador_min['id_trabajador'],
        'nombre_trabajador': trabajador_min['nombre'],
        'piso': luminaria['piso'],
        'area': luminaria['area'],
        'sector': luminaria['sector'],
        'tipo_led': luminaria['tipo_led'],
        'prioridad': luminaria['prioridad_mantenimiento'],
        'criticidad': luminaria['criticidad'],
        'dias_restantes': luminaria['dias_restantes_predichos'],
        'tiempo_estimado_min': luminaria['tiempo_reemplazo_min']
    })

# Crear DataFrame de asignaciones
df_asignaciones = pd.DataFrame(asignaciones)

# Resumen por trabajador
print("\n   " + "="*60)
print("   DISTRIBUCIÓN DE CARGA POR TRABAJADOR:")
print("   " + "="*60)

resumen_trabajadores = df_asignaciones.groupby(['id_trabajador', 'nombre_trabajador']).agg({
    'id_luminaria': 'count',
    'tiempo_estimado_min': 'sum'
}).reset_index()
resumen_trabajadores.columns = ['ID', 'Nombre', 'Cantidad_Luminarias', 'Tiempo_Total_Min']
resumen_trabajadores['Tiempo_Total_Hrs'] = resumen_trabajadores['Tiempo_Total_Min'] / 60
resumen_trabajadores['Dias_Trabajo'] = resumen_trabajadores['Tiempo_Total_Hrs'] / 8

for idx, trab in resumen_trabajadores.iterrows():
    print(f"\n   {trab['Nombre']}:")
    print(f"      • Luminarias asignadas: {trab['Cantidad_Luminarias']:.0f}")
    print(f"      • Tiempo total: {trab['Tiempo_Total_Hrs']:.1f} horas ({trab['Dias_Trabajo']:.1f} días)")
    
    # Sectores asignados
    sectores = df_asignaciones[df_asignaciones['id_trabajador'] == trab['ID']]['sector'].value_counts()
    print(f"      • Sectores: {', '.join([f'{s} ({c})' for s, c in sectores.items()])}")

# Estadísticas de balanceo
desviacion_carga = resumen_trabajadores['Tiempo_Total_Min'].std()
coeficiente_variacion = (desviacion_carga / resumen_trabajadores['Tiempo_Total_Min'].mean()) * 100

print("\n   " + "="*60)
print(f"   Balance de carga:")
print(f"      • Desviación estándar: {desviacion_carga:.1f} minutos")
print(f"      • Coeficiente de variación: {coeficiente_variacion:.1f}%")
print(f"      • Balance: {'EXCELENTE' if coeficiente_variacion < 10 else 'BUENO' if coeficiente_variacion < 20 else 'MEJORABLE'}")

# ============================================================================
# 5. EXPORTAR RESULTADOS
# ============================================================================

print("\n[5/5] Exportando resultados...")

# 1. Proyección de compras
df_proyeccion_compras.to_csv('/home/proyeccion_compras_focos.csv', index=False, encoding='utf-8-sig')
print(f"   ✓ proyeccion_compras_focos.csv")

# 2. Proyección por criticidad
df_criticos_periodo.to_csv('/home/proyeccion_por_criticidad.csv', index=False, encoding='utf-8-sig')
print(f"   ✓ proyeccion_por_criticidad.csv")

# 3. Asignaciones de trabajo
df_asignaciones.to_csv('/home/asignacion_trabajadores.csv', index=False, encoding='utf-8-sig')
print(f"   ✓ asignacion_trabajadores.csv")

# 4. Resumen por trabajador
resumen_trabajadores.to_csv('/home/resumen_trabajadores.csv', index=False, encoding='utf-8-sig')
print(f"   ✓ resumen_trabajadores.csv")

# 5. Excel consolidado
with pd.ExcelWriter('/home/Planificacion_Operativa.xlsx', engine='openpyxl') as writer:
    df_proyeccion_compras.to_excel(writer, sheet_name='Proyeccion_Compras', index=False)
    df_criticos_periodo.to_excel(writer, sheet_name='Por_Criticidad', index=False)
    df_asignaciones.to_excel(writer, sheet_name='Asignacion_Trabajo', index=False)
    resumen_trabajadores.to_excel(writer, sheet_name='Resumen_Trabajadores', index=False)

print(f"   ✓ Planificacion_Operativa.xlsx (consolidado)")

# ============================================================================
# 6. RESUMEN EJECUTIVO
# ============================================================================

print("\n" + "="*80)
print("RESUMEN EJECUTIVO - PLANIFICACIÓN OPERATIVA")
print("="*80)

print("\n📦 PROYECCIÓN DE COMPRAS:")
print(f"   Próximos 30 días:  {len(df[df['dias_restantes_predichos'] <= 30]):,} focos")
print(f"   Próximos 90 días:  {len(df[df['dias_restantes_predichos'] <= 90]):,} focos")
print(f"   Próximo año:       {len(df[df['dias_restantes_predichos'] <= 365]):,} focos")

print("\n💰 INVERSIÓN ESTIMADA:")
inv_30 = df_proyeccion_compras[df_proyeccion_compras['dias'] == 30]['costo_total_clp'].sum()
inv_90 = df_proyeccion_compras[df_proyeccion_compras['dias'] == 90]['costo_total_clp'].sum()
inv_365 = df_proyeccion_compras[df_proyeccion_compras['dias'] == 365]['costo_total_clp'].sum()

print(f"   Próximos 30 días:  ${inv_30:,.0f} CLP")
print(f"   Próximos 90 días:  ${inv_90:,.0f} CLP")
print(f"   Próximo año:       ${inv_365:,.0f} CLP")

print("\n👷 CARGA DE TRABAJO (próximos 90 días):")
print(f"   Total luminarias:  {len(df_asignar):,}")
print(f"   Horas totales:     {(df_asignar['tiempo_reemplazo_min'].sum())/60:,.1f} hrs")
print(f"   Trabajadores:      {NUM_TRABAJADORES}")
print(f"   Balance:           {coeficiente_variacion:.1f}% variación")

print("\n⚠️ ÁREAS CRÍTICAS CON PRIORIDAD 1:")
criticas_p1 = df_asignar[(df_asignar['criticidad'] == 'CRÍTICA') & 
                          (df_asignar['prioridad_mantenimiento'] == 1)]
print(f"   Total: {len(criticas_p1):,} luminarias")
if len(criticas_p1) > 0:
    areas_criticas = criticas_p1['area'].value_counts().head(5)
    for area, count in areas_criticas.items():
        print(f"   • {area}: {count} luminarias")

print("\n" + "="*80)
print("✅ PLANIFICACIÓN OPERATIVA COMPLETADA")
print("="*80)
print("\nArchivos generados:")
print("  • proyeccion_compras_focos.csv")
print("  • proyeccion_por_criticidad.csv")
print("  • asignacion_trabajadores.csv")
print("  • resumen_trabajadores.csv")
print("  • Planificacion_Operativa.xlsx (consolidado)")
print("="*80)
