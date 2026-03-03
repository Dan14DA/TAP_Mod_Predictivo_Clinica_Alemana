"""
Generación de Datos Simulados - Clínica Alemana
Mantenimiento Predictivo de Iluminación LED
Autor: Trabajo de Aplicación Práctica - Data Science
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configurar semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

print("="*70)
print("GENERACIÓN DE DATOS SIMULADOS - CLÍNICA ALEMANA")
print("Sistema de Mantenimiento Predictivo de Iluminación LED")
print("="*70)

# ============================================================================
# 1. DEFINICIÓN DE ESTRUCTURA DE LA CLÍNICA
# ============================================================================

# Definición de pisos y áreas según estructura real
areas_clinica = {
    # SUBTERRÁNEOS
    'Piso -2': [
        {'area': 'Estacionamiento', 'cantidad': 150, 'criticidad': 'BAJA', 'horas_dia': 24},
        {'area': 'Bodegas', 'cantidad': 50, 'criticidad': 'BAJA', 'horas_dia': 8}
    ],
    'Piso -1': [
        {'area': 'Estacionamiento', 'cantidad': 150, 'criticidad': 'BAJA', 'horas_dia': 24},
        {'area': 'Servicios Generales', 'cantidad': 80, 'criticidad': 'MEDIA', 'horas_dia': 24}
    ],
    
    # PISO 1 - URGENCIAS
    'Piso 1': [
        {'area': 'Urgencias General', 'cantidad': 120, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Urgencias Escolar', 'cantidad': 60, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Hall Principal', 'cantidad': 80, 'criticidad': 'ALTA', 'horas_dia': 24},
        {'area': 'Admisiones', 'cantidad': 40, 'criticidad': 'ALTA', 'horas_dia': 16}
    ],
    
    # PISO 2 - DIAGNÓSTICO
    'Piso 2': [
        {'area': 'Imagenología', 'cantidad': 70, 'criticidad': 'ALTA', 'horas_dia': 16},
        {'area': 'Laboratorio', 'cantidad': 80, 'criticidad': 'ALTA', 'horas_dia': 24},
        {'area': 'Centro Consultas', 'cantidad': 90, 'criticidad': 'MEDIA', 'horas_dia': 12},
        {'area': 'Pasillos Principales', 'cantidad': 50, 'criticidad': 'ALTA', 'horas_dia': 24}
    ],
    
    # PISO 3 - PROCEDIMIENTOS
    'Piso 3': [
        {'area': 'Salas Procedimientos', 'cantidad': 100, 'criticidad': 'ALTA', 'horas_dia': 12},
        {'area': 'Oratorio', 'cantidad': 30, 'criticidad': 'MEDIA', 'horas_dia': 12},
        {'area': 'Pasillos', 'cantidad': 40, 'criticidad': 'MEDIA', 'horas_dia': 24}
    ],
    
    # PISOS 4-9 - HOSPITALIZACIÓN (6 pisos similares)
    'Piso 4': [
        {'area': 'Habitaciones Pacientes', 'cantidad': 120, 'criticidad': 'ALTA', 'horas_dia': 24},
        {'area': 'Estación Enfermería', 'cantidad': 30, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Pasillos', 'cantidad': 35, 'criticidad': 'ALTA', 'horas_dia': 24}
    ],
    'Piso 5': [
        {'area': 'Habitaciones Pacientes', 'cantidad': 120, 'criticidad': 'ALTA', 'horas_dia': 24},
        {'area': 'Estación Enfermería', 'cantidad': 30, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Pasillos', 'cantidad': 35, 'criticidad': 'ALTA', 'horas_dia': 24}
    ],
    'Piso 6': [
        {'area': 'Habitaciones Pacientes', 'cantidad': 120, 'criticidad': 'ALTA', 'horas_dia': 24},
        {'area': 'Estación Enfermería', 'cantidad': 30, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Pasillos', 'cantidad': 35, 'criticidad': 'ALTA', 'horas_dia': 24}
    ],
    'Piso 7': [
        {'area': 'Habitaciones Pacientes', 'cantidad': 120, 'criticidad': 'ALTA', 'horas_dia': 24},
        {'area': 'Estación Enfermería', 'cantidad': 30, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Pasillos', 'cantidad': 35, 'criticidad': 'ALTA', 'horas_dia': 24}
    ],
    'Piso 8': [
        {'area': 'Habitaciones Pacientes', 'cantidad': 120, 'criticidad': 'ALTA', 'horas_dia': 24},
        {'area': 'Estación Enfermería', 'cantidad': 30, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Pasillos', 'cantidad': 35, 'criticidad': 'ALTA', 'horas_dia': 24}
    ],
    'Piso 9': [
        {'area': 'Habitaciones Pacientes', 'cantidad': 120, 'criticidad': 'ALTA', 'horas_dia': 24},
        {'area': 'Estación Enfermería', 'cantidad': 30, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Pasillos', 'cantidad': 35, 'criticidad': 'ALTA', 'horas_dia': 24}
    ],
    
    # PISOS 10-12 - ÁREA QUIRÚRGICA
    'Piso 10': [
        {'area': 'Pabellón Quirúrgico', 'cantidad': 80, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Pre-Operatorio', 'cantidad': 40, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Pasillos Quirófano', 'cantidad': 30, 'criticidad': 'CRÍTICA', 'horas_dia': 24}
    ],
    'Piso 11': [
        {'area': 'Pabellón Quirúrgico', 'cantidad': 80, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'UCI', 'cantidad': 90, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Recuperación', 'cantidad': 50, 'criticidad': 'CRÍTICA', 'horas_dia': 24}
    ],
    'Piso 12': [
        {'area': 'Pabellón Quirúrgico', 'cantidad': 60, 'criticidad': 'CRÍTICA', 'horas_dia': 24},
        {'area': 'Esterilización', 'cantidad': 40, 'criticidad': 'ALTA', 'horas_dia': 24},
        {'area': 'Sala Apoyo', 'cantidad': 30, 'criticidad': 'ALTA', 'horas_dia': 24}
    ],
    
    # PISOS 13-16 - CONSULTAS
    'Piso 13': [
        {'area': 'Consultorios Médicos', 'cantidad': 110, 'criticidad': 'MEDIA', 'horas_dia': 12},
        {'area': 'Sala Espera', 'cantidad': 35, 'criticidad': 'MEDIA', 'horas_dia': 12}
    ],
    'Piso 14': [
        {'area': 'Consultorios Médicos', 'cantidad': 110, 'criticidad': 'MEDIA', 'horas_dia': 12},
        {'area': 'Sala Espera', 'cantidad': 35, 'criticidad': 'MEDIA', 'horas_dia': 12}
    ],
    'Piso 15': [
        {'area': 'Consultorios Médicos', 'cantidad': 110, 'criticidad': 'MEDIA', 'horas_dia': 12},
        {'area': 'Sala Espera', 'cantidad': 35, 'criticidad': 'MEDIA', 'horas_dia': 12}
    ],
    'Piso 16': [
        {'area': 'Oficinas Administrativas', 'cantidad': 90, 'criticidad': 'BAJA', 'horas_dia': 10},
        {'area': 'Sala Reuniones', 'cantidad': 40, 'criticidad': 'BAJA', 'horas_dia': 10},
        {'area': 'Cafetería Staff', 'cantidad': 30, 'criticidad': 'MEDIA', 'horas_dia': 14}
    ]
}

# ============================================================================
# 2. TIPOS DE FOCOS LED Y ESPECIFICACIONES
# ============================================================================

tipos_led = {
    'LED Panel 60x60': {
        'vida_util_nominal': 50000,  # horas
        'potencia': 40,  # watts
        'uso_principal': ['Consultorios Médicos', 'Oficinas Administrativas', 'Sala Reuniones'],
        'variacion_vida': 0.15  # ±15% variación
    },
    'LED Tubo T8': {
        'vida_util_nominal': 40000,
        'potencia': 18,
        'uso_principal': ['Pasillos', 'Estacionamiento', 'Bodegas'],
        'variacion_vida': 0.18
    },
    'LED Downlight': {
        'vida_util_nominal': 45000,
        'potencia': 12,
        'uso_principal': ['Habitaciones Pacientes', 'Sala Espera', 'Hall Principal'],
        'variacion_vida': 0.12
    },
    'LED Quirúrgico': {
        'vida_util_nominal': 60000,
        'potencia': 50,
        'uso_principal': ['Pabellón Quirúrgico', 'UCI'],
        'variacion_vida': 0.10  # Menor variación, mejor calidad
    },
    'LED Emergencia': {
        'vida_util_nominal': 55000,
        'potencia': 8,
        'uso_principal': ['Estación Enfermería', 'Urgencias General', 'Urgencias Escolar'],
        'variacion_vida': 0.12
    },
    'LED Industrial': {
        'vida_util_nominal': 50000,
        'potencia': 60,
        'uso_principal': ['Laboratorio', 'Imagenología', 'Esterilización'],
        'variacion_vida': 0.14
    }
}

# Marcas y sus factores de calidad (afecta vida útil real)
marcas_led = {
    'Philips': 1.10,      # +10% vida útil
    'Osram': 1.08,        # +8%
    'Samsung': 1.05,      # +5%
    'LG': 1.03,           # +3%
    'Genérico A': 0.95,   # -5%
    'Genérico B': 0.90    # -10%
}

# ============================================================================
# 3. GENERACIÓN DE INVENTARIO DE LUMINARIAS
# ============================================================================

print("\n[1/5] Generando inventario de luminarias...")

luminarias = []
id_counter = 1

for piso, areas in areas_clinica.items():
    for area_info in areas:
        area_nombre = area_info['area']
        cantidad = area_info['cantidad']
        criticidad = area_info['criticidad']
        horas_dia = area_info['horas_dia']
        
        # Determinar tipo de LED según área
        tipo_led = None
        for tipo, specs in tipos_led.items():
            if area_nombre in specs['uso_principal']:
                tipo_led = tipo
                break
        
        # Si no hay match específico, asignar por defecto
        if tipo_led is None:
            if 'Quirúrgico' in area_nombre or 'UCI' in area_nombre:
                tipo_led = 'LED Quirúrgico'
            elif 'Urgencias' in area_nombre or 'Enfermería' in area_nombre:
                tipo_led = 'LED Emergencia'
            elif 'Consultorio' in area_nombre or 'Oficina' in area_nombre:
                tipo_led = 'LED Panel 60x60'
            elif 'Pasillo' in area_nombre:
                tipo_led = 'LED Tubo T8'
            else:
                tipo_led = 'LED Downlight'
        
        # Generar luminarias para esta área
        for i in range(cantidad):
            # Asignar marca (distribución realista)
            marca = random.choices(
                list(marcas_led.keys()),
                weights=[25, 20, 15, 15, 15, 10],  # Philips y Osram más comunes
                k=1
            )[0]
            
            # Fecha de instalación (últimos 3-5 años, concentrado en 2022-2023)
            dias_desde_instalacion = random.randint(365, 1825)  # 1-5 años
            fecha_instalacion = datetime(2024, 8, 1) - timedelta(days=dias_desde_instalacion)
            
            luminaria = {
                'id_luminaria': f'LED-{id_counter:05d}',
                'piso': piso,
                'area': area_nombre,
                'ubicacion_especifica': f'{area_nombre} - Sector {(i//10)+1}',
                'tipo_led': tipo_led,
                'marca': marca,
                'potencia_watts': tipos_led[tipo_led]['potencia'],
                'vida_util_nominal': tipos_led[tipo_led]['vida_util_nominal'],
                'fecha_instalacion': fecha_instalacion,
                'criticidad': criticidad,
                'horas_operacion_dia': horas_dia,
                'estado_actual': 'OPERATIVO'  # Todos parten operativos
            }
            
            luminarias.append(luminaria)
            id_counter += 1

df_luminarias = pd.DataFrame(luminarias)

print(f"   ✓ Total luminarias generadas: {len(df_luminarias):,}")
print(f"   ✓ Distribución por criticidad:")
for crit in ['CRÍTICA', 'ALTA', 'MEDIA', 'BAJA']:
    count = len(df_luminarias[df_luminarias['criticidad'] == crit])
    pct = (count/len(df_luminarias))*100
    print(f"     • {crit}: {count:,} ({pct:.1f}%)")

# ============================================================================
# 4. GENERACIÓN DE HISTORIAL DE OPERACIÓN Y FALLAS
# ============================================================================

print("\n[2/5] Generando historial de operación (18 meses)...")

# Período de simulación: Agosto 2024 a Febrero 2026
fecha_inicio = datetime(2024, 8, 1)
fecha_fin = datetime(2026, 2, 11)  # Fecha actual
dias_simulacion = (fecha_fin - fecha_inicio).days

registros_operacion = []

for idx, lum in df_luminarias.iterrows():
    dias_instalado = (fecha_fin - lum['fecha_instalacion']).days
    horas_totales_uso = dias_instalado * lum['horas_operacion_dia']
    
    # Calcular vida útil real considerando factores
    vida_nominal = lum['vida_util_nominal']
    factor_marca = marcas_led[lum['marca']]
    variacion_aleatoria = np.random.normal(1.0, tipos_led[lum['tipo_led']]['variacion_vida'])
    
    vida_util_real = vida_nominal * factor_marca * variacion_aleatoria
    
    # Factores ambientales según área
    if 'Quirúrgico' in lum['area'] or 'UCI' in lum['area']:
        factor_ambiental = np.random.uniform(0.98, 1.02)  # Ambiente controlado
        ciclos_dia = 5  # Pocos encendidos
    elif 'Estacionamiento' in lum['area']:
        factor_ambiental = np.random.uniform(0.85, 0.95)  # Humedad, temperatura
        ciclos_dia = 2
    elif 'Urgencias' in lum['area']:
        factor_ambiental = np.random.uniform(0.90, 1.00)
        ciclos_dia = 10  # Muchos encendidos
    elif 'Consultorio' in lum['area']:
        factor_ambiental = np.random.uniform(0.92, 1.00)
        ciclos_dia = 8
    else:
        factor_ambiental = np.random.uniform(0.90, 1.00)
        ciclos_dia = 6
    
    vida_util_real *= factor_ambiental
    
    # Calcular si ha fallado
    porcentaje_vida_usado = (horas_totales_uso / vida_util_real) * 100
    
    # Probabilidad de falla aumenta exponencialmente después del 80% de vida útil
    if porcentaje_vida_usado > 80:
        prob_falla = min(0.95, (porcentaje_vida_usado - 80) / 20 * 0.8)
    else:
        prob_falla = 0.02  # Falla prematura muy baja
    
    ha_fallado = np.random.random() < prob_falla
    
    if ha_fallado:
        # Calcular fecha de falla
        horas_hasta_falla = vida_util_real * np.random.uniform(0.85, 0.98)
        dias_hasta_falla = int(horas_hasta_falla / lum['horas_operacion_dia'])
        fecha_falla = lum['fecha_instalacion'] + timedelta(days=dias_hasta_falla)
        
        if fecha_falla <= fecha_fin:
            estado = 'REEMPLAZADO'
            fecha_cambio = fecha_falla + timedelta(days=random.randint(1, 7))
            motivo = 'FALLA'
        else:
            estado = 'OPERATIVO'
            fecha_falla = None
            fecha_cambio = None
            motivo = None
    else:
        estado = 'OPERATIVO'
        fecha_falla = None
        fecha_cambio = None
        motivo = None
    
    registro = {
        'id_luminaria': lum['id_luminaria'],
        'vida_util_real_calculada': int(vida_util_real),
        'horas_uso_acumuladas': int(horas_totales_uso),
        'porcentaje_vida_consumida': round(porcentaje_vida_usado, 2),
        'ciclos_encendido_dia': ciclos_dia,
        'ciclos_acumulados': dias_instalado * ciclos_dia,
        'temperatura_promedio_c': np.random.uniform(18, 26),
        'humedad_promedio_pct': np.random.uniform(40, 70),
        'voltaje_promedio_v': np.random.uniform(218, 224),
        'fecha_falla': fecha_falla,
        'fecha_cambio': fecha_cambio,
        'motivo_cambio': motivo,
        'estado_actual': estado
    }
    
    registros_operacion.append(registro)

df_operacion = pd.DataFrame(registros_operacion)

# Combinar dataframes
df_completo = pd.merge(df_luminarias, df_operacion, on='id_luminaria')

# Actualizar estado
df_completo['estado_actual'] = df_completo['estado_actual_y']
df_completo = df_completo.drop(['estado_actual_x', 'estado_actual_y'], axis=1)

print(f"   ✓ Registros generados: {len(df_completo):,}")
print(f"   ✓ Luminarias que han fallado: {len(df_completo[df_completo['estado_actual'] == 'REEMPLAZADO']):,}")
print(f"   ✓ Luminarias operativas: {len(df_completo[df_completo['estado_actual'] == 'OPERATIVO']):,}")

# ============================================================================
# 5. CÁLCULO DE MÉTRICAS ADICIONALES
# ============================================================================

print("\n[3/5] Calculando métricas adicionales...")

df_completo['dias_desde_instalacion'] = (fecha_fin - df_completo['fecha_instalacion']).dt.days
df_completo['horas_restantes_estimadas'] = df_completo['vida_util_real_calculada'] - df_completo['horas_uso_acumuladas']
df_completo['dias_restantes_estimados'] = (df_completo['horas_restantes_estimadas'] / df_completo['horas_operacion_dia']).round(0)

# Clasificación de riesgo
def clasificar_riesgo(row):
    if row['estado_actual'] == 'REEMPLAZADO':
        return 'REEMPLAZADO'
    
    pct_vida = row['porcentaje_vida_consumida']
    criticidad = row['criticidad']
    
    if pct_vida >= 90:
        if criticidad in ['CRÍTICA', 'ALTA']:
            return 'MUY ALTO'
        else:
            return 'ALTO'
    elif pct_vida >= 80:
        if criticidad == 'CRÍTICA':
            return 'ALTO'
        else:
            return 'MEDIO'
    elif pct_vida >= 70:
        return 'MEDIO'
    else:
        return 'BAJO'

df_completo['nivel_riesgo'] = df_completo.apply(clasificar_riesgo, axis=1)

# Prioridad de mantenimiento (1-5, siendo 1 más urgente)
def calcular_prioridad(row):
    if row['estado_actual'] == 'REEMPLAZADO':
        return 0  # Ya reemplazado
    
    pct_vida = row['porcentaje_vida_consumida']
    criticidad = row['criticidad']
    
    score = pct_vida / 100  # Base
    
    # Ajuste por criticidad
    if criticidad == 'CRÍTICA':
        score *= 1.5
    elif criticidad == 'ALTA':
        score *= 1.3
    elif criticidad == 'MEDIA':
        score *= 1.1
    
    if score >= 1.2:
        return 1
    elif score >= 1.0:
        return 2
    elif score >= 0.8:
        return 3
    elif score >= 0.6:
        return 4
    else:
        return 5

df_completo['prioridad_mantenimiento'] = df_completo.apply(calcular_prioridad, axis=1)

print(f"   ✓ Métricas calculadas exitosamente")
print(f"   ✓ Distribución de riesgo:")
for riesgo in ['MUY ALTO', 'ALTO', 'MEDIO', 'BAJO', 'REEMPLAZADO']:
    count = len(df_completo[df_completo['nivel_riesgo'] == riesgo])
    pct = (count/len(df_completo))*100
    print(f"     • {riesgo}: {count:,} ({pct:.1f}%)")

# ============================================================================
# 6. GENERACIÓN DE TABLA DE COSTOS
# ============================================================================

print("\n[4/5] Generando tabla de costos...")

costos = {
    'tipo_led': list(tipos_led.keys()),
    'costo_unidad_clp': [12990, 8990, 9990, 45990, 15990, 18990],
    'costo_instalacion_clp': [5000, 4000, 4500, 15000, 6000, 7000],
    'tiempo_instalacion_min': [15, 12, 15, 45, 20, 25]
}

df_costos = pd.DataFrame(costos)

# Calcular costos de mantenimiento reactivo vs preventivo
costo_emergencia_multiplicador = 2.5  # Mantenimiento reactivo cuesta 2.5x más

print(f"   ✓ Tabla de costos generada")

# ============================================================================
# 7. EXPORTAR DATOS
# ============================================================================

print("\n[5/5] Exportando archivos...")

# Guardar CSV principal
df_completo.to_csv('/home/claude/datos_luminarias_clinica_alemana.csv', index=False, encoding='utf-8-sig')
print(f"   ✓ Archivo principal: datos_luminarias_clinica_alemana.csv")

# Guardar Excel con múltiples hojas
with pd.ExcelWriter('/home/claude/datos_luminarias_clinica_alemana.xlsx', engine='openpyxl') as writer:
    df_completo.to_excel(writer, sheet_name='Luminarias_Completo', index=False)
    df_costos.to_excel(writer, sheet_name='Costos', index=False)
    
    # Resumen por piso
    resumen_piso = df_completo.groupby('piso').agg({
        'id_luminaria': 'count',
        'horas_uso_acumuladas': 'mean',
        'porcentaje_vida_consumida': 'mean',
        'vida_util_real_calculada': 'mean'
    }).round(2)
    resumen_piso.columns = ['Cantidad', 'Horas_Uso_Promedio', 'Pct_Vida_Promedio', 'Vida_Util_Real_Promedio']
    resumen_piso.to_excel(writer, sheet_name='Resumen_Pisos')
    
    # Resumen por área crítica
    resumen_criticidad = df_completo.groupby('criticidad').agg({
        'id_luminaria': 'count',
        'porcentaje_vida_consumida': 'mean'
    }).round(2)
    resumen_criticidad.columns = ['Cantidad', 'Pct_Vida_Promedio']
    resumen_criticidad.to_excel(writer, sheet_name='Resumen_Criticidad')

print(f"   ✓ Archivo Excel: datos_luminarias_clinica_alemana.xlsx")

# ============================================================================
# 8. ESTADÍSTICAS FINALES
# ============================================================================

print("\n" + "="*70)
print("RESUMEN DE DATOS GENERADOS")
print("="*70)
print(f"Total luminarias:              {len(df_completo):,}")
print(f"Período simulado:              18 meses (Ago 2024 - Feb 2026)")
print(f"Rango fechas instalación:      {df_completo['fecha_instalacion'].min().date()} a {df_completo['fecha_instalacion'].max().date()}")
print(f"\nLuminarias por estado:")
print(f"  • Operativas:                {len(df_completo[df_completo['estado_actual'] == 'OPERATIVO']):,}")
print(f"  • Reemplazadas:              {len(df_completo[df_completo['estado_actual'] == 'REEMPLAZADO']):,}")
print(f"\nLuminarias en riesgo ALTO/MUY ALTO:")
alto_riesgo = len(df_completo[df_completo['nivel_riesgo'].isin(['ALTO', 'MUY ALTO'])])
print(f"  • Total:                     {alto_riesgo:,}")
print(f"  • % del total:               {(alto_riesgo/len(df_completo)*100):.1f}%")
print(f"\nPrioridad 1 (más urgente):")
prioridad_1 = len(df_completo[df_completo['prioridad_mantenimiento'] == 1])
print(f"  • Total:                     {prioridad_1:,}")
print(f"  • % del total:               {(prioridad_1/len(df_completo)*100):.1f}%")
print(f"\nVida útil promedio:")
print(f"  • Horas uso acumuladas:      {df_completo['horas_uso_acumuladas'].mean():,.0f}")
print(f"  • % vida consumida:          {df_completo['porcentaje_vida_consumida'].mean():.1f}%")
print("="*70)
print("\n✅ GENERACIÓN DE DATOS COMPLETADA EXITOSAMENTE\n")
