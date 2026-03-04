"""
Análisis Exploratorio de Datos (EDA)
Mantenimiento Predictivo de Iluminación LED - Clínica Alemana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("Mantenimiento Predictivo de Iluminación LED - Clínica Alemana")
print("="*80)

# ============================================================================
# 1. CARGA Y VALIDACIÓN DE DATOS (Cambiar ruta Local)
# ============================================================================

print("\n[1/7] Cargando datos...")
df = pd.read_csv('/home/datos_luminarias_clinica_alemana.csv', 
                 parse_dates=['fecha_instalacion', 'fecha_falla', 'fecha_cambio'])

print(f"   ✓ Datos cargados: {df.shape[0]:,} luminarias, {df.shape[1]} variables")
print(f"   ✓ Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Verificar datos faltantes
print(f"\n   Datos faltantes por columna:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, count in missing.items():
    pct = (count/len(df))*100
    print(f"     • {col}: {count:,} ({pct:.1f}%)")

# ============================================================================
# 2. ANÁLISIS DESCRIPTIVO GENERAL
# ============================================================================

print("\n[2/7] Estadísticas descriptivas...")

# Variables numéricas clave
vars_numericas = ['horas_uso_acumuladas', 'porcentaje_vida_consumida', 
                  'vida_util_real_calculada', 'dias_restantes_estimados']

print(f"\n   Estadísticas de variables clave:")
print(df[vars_numericas].describe().round(2).to_string())

# Distribución por categorías
print(f"\n   Distribución por Estado:")
print(df['estado_actual'].value_counts().to_string())

print(f"\n   Distribución por Nivel de Riesgo:")
print(df['nivel_riesgo'].value_counts().to_string())

print(f"\n   Distribución por Criticidad:")
print(df['criticidad'].value_counts().to_string())

# ============================================================================
# 3. ANÁLISIS POR TIPO DE LED
# ============================================================================

print("\n[3/7] Análisis por tipo de LED...")

analisis_tipo = df.groupby('tipo_led').agg({
    'id_luminaria': 'count',
    'vida_util_nominal': 'first',
    'vida_util_real_calculada': 'mean',
    'porcentaje_vida_consumida': 'mean',
    'horas_uso_acumuladas': 'mean'
}).round(2)

analisis_tipo.columns = ['Cantidad', 'Vida_Nominal', 'Vida_Real_Promedio', 
                         'Pct_Vida_Usado', 'Horas_Uso_Promedio']

# Calcular tasa de falla
fallas_por_tipo = df[df['estado_actual'] == 'REEMPLAZADO'].groupby('tipo_led').size()
total_por_tipo = df.groupby('tipo_led').size()
tasa_falla = (fallas_por_tipo / total_por_tipo * 100).round(2)
analisis_tipo['Tasa_Falla_%'] = tasa_falla

print(analisis_tipo.to_string())

# ============================================================================
# 4. ANÁLISIS POR MARCA
# ============================================================================

print("\n[4/7] Análisis por marca...")

analisis_marca = df.groupby('marca').agg({
    'id_luminaria': 'count',
    'vida_util_real_calculada': 'mean',
    'porcentaje_vida_consumida': 'mean'
}).round(2)

analisis_marca.columns = ['Cantidad', 'Vida_Real_Promedio', 'Pct_Vida_Usado']

# Tasa de falla por marca
fallas_por_marca = df[df['estado_actual'] == 'REEMPLAZADO'].groupby('marca').size()
total_por_marca = df.groupby('marca').size()
tasa_falla_marca = (fallas_por_marca / total_por_marca * 100).round(2)
analisis_marca['Tasa_Falla_%'] = tasa_falla_marca
analisis_marca = analisis_marca.sort_values('Tasa_Falla_%')

print(analisis_marca.to_string())

# ============================================================================
# 5. ANÁLISIS POR CRITICIDAD Y ÁREA
# ============================================================================

print("\n[5/7] Análisis por criticidad...")

# Luminarias operativas por criticidad
operativas = df[df['estado_actual'] == 'OPERATIVO']

riesgo_por_criticidad = pd.crosstab(
    operativas['criticidad'], 
    operativas['nivel_riesgo'],
    margins=True,
    margins_name='TOTAL'
)

print(f"\n   Distribución de riesgo por criticidad (solo operativas):")
print(riesgo_por_criticidad.to_string())

# Áreas con mayor riesgo
print(f"\n   Top 10 áreas con más luminarias en riesgo ALTO/MUY ALTO:")
alto_riesgo = operativas[operativas['nivel_riesgo'].isin(['ALTO', 'MUY ALTO'])]
top_areas_riesgo = alto_riesgo['area'].value_counts().head(10)
for area, count in top_areas_riesgo.items():
    print(f"     • {area}: {count}")

# ============================================================================
# 6. ANÁLISIS TEMPORAL DE FALLAS
# ============================================================================

print("\n[6/7] Análisis temporal de fallas...")

fallas = df[df['estado_actual'] == 'REEMPLAZADO'].copy()
fallas['mes_falla'] = fallas['fecha_falla'].dt.to_period('M')

fallas_por_mes = fallas.groupby('mes_falla').size().reset_index()
fallas_por_mes.columns = ['Mes', 'Cantidad_Fallas']

print(f"\n   Fallas por mes (últimos 12 meses):")
print(fallas_por_mes.tail(12).to_string(index=False))

# Promedio de días entre instalación y falla
dias_hasta_falla = (fallas['fecha_falla'] - fallas['fecha_instalacion']).dt.days
print(f"\n   Tiempo promedio hasta falla: {dias_hasta_falla.mean():.0f} días ({dias_hasta_falla.mean()/365:.1f} años)")
print(f"   Desviación estándar: {dias_hasta_falla.std():.0f} días")
print(f"   Mínimo: {dias_hasta_falla.min()} días")
print(f"   Máximo: {dias_hasta_falla.max()} días")

# ============================================================================
# 7. ANÁLISIS DE PRIORIDADES DE MANTENIMIENTO
# ============================================================================

print("\n[7/7] Análisis de prioridades de mantenimiento...")

# Solo luminarias operativas
prioridades = operativas.groupby(['prioridad_mantenimiento', 'criticidad']).size().reset_index()
prioridades.columns = ['Prioridad', 'Criticidad', 'Cantidad']

print(f"\n   Distribución de prioridades (luminarias operativas):")
pivot_prioridad = prioridades.pivot_table(
    index='Prioridad',
    columns='Criticidad',
    values='Cantidad',
    fill_value=0
).astype(int)
print(pivot_prioridad.to_string())

# Resumen de mantenimiento urgente
print(f"\n   RESUMEN DE MANTENIMIENTO URGENTE:")
prioridad_1 = operativas[operativas['prioridad_mantenimiento'] == 1]
print(f"     • Total Prioridad 1: {len(prioridad_1):,}")
print(f"     • En áreas CRÍTICAS: {len(prioridad_1[prioridad_1['criticidad'] == 'CRÍTICA']):,}")
print(f"     • En áreas ALTAS: {len(prioridad_1[prioridad_1['criticidad'] == 'ALTA']):,}")

# Días promedio restantes para prioridad 1
dias_rest_p1 = prioridad_1['dias_restantes_estimados'].mean()
print(f"     • Días restantes promedio: {dias_rest_p1:.0f} días")

# ============================================================================
# 8. GENERACIÓN DE VISUALIZACIONES
# ============================================================================

print("\n[8/7] Generando visualizaciones...")

# Crear figura con múltiples subplots
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Análisis Exploratorio - Mantenimiento Predictivo Iluminación LED\nClínica Alemana', 
             fontsize=16, fontweight='bold')

# 1. Distribución de % vida consumida
axes[0, 0].hist(df['porcentaje_vida_consumida'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['porcentaje_vida_consumida'].mean(), color='red', 
                   linestyle='--', label=f'Media: {df["porcentaje_vida_consumida"].mean():.1f}%')
axes[0, 0].set_xlabel('% Vida Consumida')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Distribución de Vida Útil Consumida')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Estado actual
estado_counts = df['estado_actual'].value_counts()
axes[0, 1].bar(estado_counts.index, estado_counts.values, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Estado')
axes[0, 1].set_ylabel('Cantidad')
axes[0, 1].set_title('Distribución por Estado')
axes[0, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(estado_counts.values):
    axes[0, 1].text(i, v + 50, str(v), ha='center', fontweight='bold')

# 3. Nivel de riesgo
riesgo_counts = df['nivel_riesgo'].value_counts()
colors_riesgo = {'MUY ALTO': '#d62728', 'ALTO': '#ff7f0e', 'MEDIO': '#ffbb00', 
                 'BAJO': '#2ca02c', 'REEMPLAZADO': '#7f7f7f'}
colors_plot = [colors_riesgo.get(x, '#1f77b4') for x in riesgo_counts.index]
axes[0, 2].bar(riesgo_counts.index, riesgo_counts.values, color=colors_plot, 
               edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('Nivel de Riesgo')
axes[0, 2].set_ylabel('Cantidad')
axes[0, 2].set_title('Distribución por Nivel de Riesgo')
axes[0, 2].tick_params(axis='x', rotation=45)
for i, v in enumerate(riesgo_counts.values):
    axes[0, 2].text(i, v + 30, str(v), ha='center', fontweight='bold')

# 4. Vida útil real por tipo de LED
tipo_vida = df.groupby('tipo_led')['vida_util_real_calculada'].mean().sort_values()
axes[1, 0].barh(range(len(tipo_vida)), tipo_vida.values, edgecolor='black', alpha=0.7)
axes[1, 0].set_yticks(range(len(tipo_vida)))
axes[1, 0].set_yticklabels(tipo_vida.index)
axes[1, 0].set_xlabel('Horas')
axes[1, 0].set_title('Vida Útil Real Promedio por Tipo')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 5. Tasa de falla por marca
axes[1, 1].barh(analisis_marca.index, analisis_marca['Tasa_Falla_%'], 
                edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Tasa de Falla (%)')
axes[1, 1].set_title('Tasa de Falla por Marca')
axes[1, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(analisis_marca['Tasa_Falla_%'].values):
    axes[1, 1].text(v + 0.5, i, f'{v:.1f}%', va='center')

# 6. Criticidad vs Riesgo (operativas)
pivot_heatmap = pd.crosstab(operativas['criticidad'], operativas['nivel_riesgo'])
sns.heatmap(pivot_heatmap, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 2], 
            cbar_kws={'label': 'Cantidad'})
axes[1, 2].set_title('Criticidad vs Nivel de Riesgo (Operativas)')
axes[1, 2].set_xlabel('Nivel de Riesgo')
axes[1, 2].set_ylabel('Criticidad')

# 7. Horas de uso por criticidad (boxplot)
df_plot = df[['criticidad', 'horas_uso_acumuladas']].copy()
criticidad_order = ['CRÍTICA', 'ALTA', 'MEDIA', 'BAJA']
df_plot['criticidad'] = pd.Categorical(df_plot['criticidad'], categories=criticidad_order, ordered=True)
df_plot = df_plot.sort_values('criticidad')
bp = axes[2, 0].boxplot([df_plot[df_plot['criticidad'] == c]['horas_uso_acumuladas'].values 
                          for c in criticidad_order],
                         labels=criticidad_order, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)
axes[2, 0].set_ylabel('Horas Uso Acumuladas')
axes[2, 0].set_title('Distribución de Horas de Uso por Criticidad')
axes[2, 0].grid(True, alpha=0.3, axis='y')

# 8. Top 10 áreas con más luminarias
top_areas = df['area'].value_counts().head(10)
axes[2, 1].barh(range(len(top_areas)), top_areas.values, edgecolor='black', alpha=0.7)
axes[2, 1].set_yticks(range(len(top_areas)))
axes[2, 1].set_yticklabels(top_areas.index, fontsize=9)
axes[2, 1].set_xlabel('Cantidad de Luminarias')
axes[2, 1].set_title('Top 10 Áreas con Más Luminarias')
axes[2, 1].grid(True, alpha=0.3, axis='x')

# 9. Prioridad de mantenimiento (operativas)
prioridad_counts = operativas['prioridad_mantenimiento'].value_counts().sort_index()
colors_prior = ['#d62728', '#ff7f0e', '#ffbb00', '#2ca02c', '#1f77b4']
axes[2, 2].bar(prioridad_counts.index, prioridad_counts.values, 
               color=colors_prior, edgecolor='black', alpha=0.7)
axes[2, 2].set_xlabel('Prioridad (1=Más urgente)')
axes[2, 2].set_ylabel('Cantidad')
axes[2, 2].set_title('Distribución de Prioridades de Mantenimiento')
axes[2, 2].set_xticks(prioridad_counts.index)
for i, v in enumerate(prioridad_counts.values):
    axes[2, 2].text(prioridad_counts.index[i], v + 30, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/01_analisis_exploratorio.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Gráfico guardado: 01_analisis_exploratorio.png")

# ============================================================================
# 9. CORRELACIONES
# ============================================================================

print("\n[9/7] Análisis de correlaciones...")

vars_corr = ['horas_uso_acumuladas', 'porcentaje_vida_consumida', 
             'ciclos_acumulados', 'temperatura_promedio_c', 
             'humedad_promedio_pct', 'voltaje_promedio_v',
             'vida_util_real_calculada']

corr_matrix = df[vars_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Matriz de Correlación - Variables Operacionales', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/02_correlaciones.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Gráfico guardado: 02_correlaciones.png")

# ============================================================================
# 10. INSIGHTS PRINCIPALES
# ============================================================================

print("\n" + "="*80)
print("INSIGHTS PRINCIPALES DEL ANÁLISIS EXPLORATORIO")
print("="*80)

print("\n1. ESTADO GENERAL DEL SISTEMA:")
total_operativas = len(operativas)
pct_operativas = (total_operativas/len(df))*100
print(f"   • {total_operativas:,} luminarias operativas ({pct_operativas:.1f}%)")
print(f"   • {len(fallas):,} luminarias reemplazadas por falla ({100-pct_operativas:.1f}%)")
print(f"   • Vida útil promedio consumida: {df['porcentaje_vida_consumida'].mean():.1f}%")

print("\n2. SITUACIÓN DE RIESGO:")
muy_alto_riesgo = len(operativas[operativas['nivel_riesgo'] == 'MUY ALTO'])
alto_riesgo_val = len(operativas[operativas['nivel_riesgo'] == 'ALTO'])
print(f"   • {muy_alto_riesgo:,} luminarias en riesgo MUY ALTO")
print(f"   • {alto_riesgo_val:,} luminarias en riesgo ALTO")
print(f"   • Total en riesgo crítico: {muy_alto_riesgo + alto_riesgo_val:,}")

criticas_alto_riesgo = len(operativas[(operativas['nivel_riesgo'].isin(['ALTO', 'MUY ALTO'])) & 
                                      (operativas['criticidad'] == 'CRÍTICA')])
print(f"   • Luminarias CRÍTICAS en alto riesgo: {criticas_alto_riesgo:,}")

print("\n3. MANTENIMIENTO REQUERIDO:")
print(f"   • Prioridad 1 (urgente): {len(prioridad_1):,} luminarias")
prioridad_2 = len(operativas[operativas['prioridad_mantenimiento'] == 2])
print(f"   • Prioridad 2 (muy pronto): {prioridad_2:,} luminarias")
print(f"   • Total a intervenir en corto plazo: {len(prioridad_1) + prioridad_2:,}")

print("\n4. DESEMPEÑO POR MARCA:")
mejor_marca = analisis_marca.nsmallest(1, 'Tasa_Falla_%').index[0]
peor_marca = analisis_marca.nlargest(1, 'Tasa_Falla_%').index[0]
print(f"   • Mejor desempeño: {mejor_marca} ({analisis_marca.loc[mejor_marca, 'Tasa_Falla_%']:.1f}% falla)")
print(f"   • Peor desempeño: {peor_marca} ({analisis_marca.loc[peor_marca, 'Tasa_Falla_%']:.1f}% falla)")

print("\n5. TIPO DE LED MÁS CONFIABLE:")
mejor_tipo = analisis_tipo.nsmallest(1, 'Tasa_Falla_%').index[0]
print(f"   • {mejor_tipo}: {analisis_tipo.loc[mejor_tipo, 'Tasa_Falla_%']:.1f}% tasa de falla")
print(f"   • Vida útil real promedio: {analisis_tipo.loc[mejor_tipo, 'Vida_Real_Promedio']:.0f} horas")

print("\n" + "="*80)
print("\n✅ ANÁLISIS EXPLORATORIO COMPLETADO\n")
print("Archivos generados:")
print("  • 01_analisis_exploratorio.png")
print("  • 02_correlaciones.png")
print("="*80)
