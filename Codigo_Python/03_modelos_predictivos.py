"""
Modelos Predictivos para Mantenimiento de Iluminación LED
Clínica Alemana - Trabajo de Aplicación Práctica
Modelos: Regresión, Clasificación, Análisis de Supervivencia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             classification_report, confusion_matrix, accuracy_score,
                             roc_auc_score, roc_curve)
from lifelines import KaplanMeierFitter, CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODELOS PREDICTIVOS - MANTENIMIENTO ILUMINACIÓN LED")
print("Clínica Alemana")
print("="*80)

# ============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS (Cambiar ruta Local)
# ============================================================================

print("\n[1/6] Cargando y preparando datos...")

df = pd.read_csv('/home/datos_luminarias_clinica_alemana.csv',
                 parse_dates=['fecha_instalacion', 'fecha_falla', 'fecha_cambio'])

# Filtrar solo luminarias operativas para predicción
df_operativas = df[df['estado_actual'] == 'OPERATIVO'].copy()
print(f"   ✓ Dataset operativas: {len(df_operativas):,} luminarias")

# Encodear variables categóricas
le_tipo = LabelEncoder()
le_marca = LabelEncoder()
le_criticidad = LabelEncoder()
le_piso = LabelEncoder()

df_operativas['tipo_led_encoded'] = le_tipo.fit_transform(df_operativas['tipo_led'])
df_operativas['marca_encoded'] = le_marca.fit_transform(df_operativas['marca'])
df_operativas['criticidad_encoded'] = le_criticidad.fit_transform(df_operativas['criticidad'])
df_operativas['piso_encoded'] = le_piso.fit_transform(df_operativas['piso'])

# Variables predictoras
feature_cols = [
    'horas_uso_acumuladas',
    'porcentaje_vida_consumida',
    'ciclos_acumulados',
    'temperatura_promedio_c',
    'humedad_promedio_pct',
    'voltaje_promedio_v',
    'vida_util_real_calculada',
    'horas_operacion_dia',
    'tipo_led_encoded',
    'marca_encoded',
    'criticidad_encoded',
    'potencia_watts'
]

X = df_operativas[feature_cols]
print(f"   ✓ Variables predictoras: {len(feature_cols)}")

# ============================================================================
# 2. MODELO 1: REGRESIÓN - PREDICCIÓN DE DÍAS RESTANTES
# ============================================================================

print("\n[2/6] Modelo 1: Regresión - Predicción de Días Restantes...")

# Target: días restantes estimados
y_regresion = df_operativas['dias_restantes_estimados']

# Split train/test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regresion, test_size=0.2, random_state=42
)

print(f"   • Train: {len(X_train_reg):,} | Test: {len(X_test_reg):,}")

# Modelo 1a: Regresión Lineal Multiple
print("\n   a) Regresión Lineal Múltiple...")
lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)
y_pred_lr = lr_model.predict(X_test_reg)

mae_lr = mean_absolute_error(y_test_reg, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_reg, y_pred_lr))
r2_lr = r2_score(y_test_reg, y_pred_lr)

print(f"      MAE:  {mae_lr:.2f} días")
print(f"      RMSE: {rmse_lr:.2f} días")
print(f"      R²:   {r2_lr:.4f}")

# Modelo 1b: Random Forest Regressor
print("\n   b) Random Forest Regressor...")
rf_reg_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf_reg_model.fit(X_train_reg, y_train_reg)
y_pred_rf = rf_reg_model.predict(X_test_reg)

mae_rf = mean_absolute_error(y_test_reg, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf))
r2_rf = r2_score(y_test_reg, y_pred_rf)

print(f"      MAE:  {mae_rf:.2f} días")
print(f"      RMSE: {rmse_rf:.2f} días")
print(f"      R²:   {r2_rf:.4f}")

# Modelo 1c: Gradient Boosting
print("\n   c) Gradient Boosting Regressor...")
gb_reg_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_reg_model.fit(X_train_reg, y_train_reg)
y_pred_gb = gb_reg_model.predict(X_test_reg)

mae_gb = mean_absolute_error(y_test_reg, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test_reg, y_pred_gb))
r2_gb = r2_score(y_test_reg, y_pred_gb)

print(f"      MAE:  {mae_gb:.2f} días")
print(f"      RMSE: {rmse_gb:.2f} días")
print(f"      R²:   {r2_gb:.4f}")

# Seleccionar mejor modelo
best_reg_model = rf_reg_model if r2_rf > r2_gb else gb_reg_model
best_reg_name = "Random Forest" if r2_rf > r2_gb else "Gradient Boosting"
print(f"\n   ✓ Mejor modelo de regresión: {best_reg_name}")

# Importancia de características
feature_importance_reg = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_reg_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 5 variables más importantes:")
for idx, row in feature_importance_reg.head(5).iterrows():
    print(f"      {row['feature']}: {row['importance']:.4f}")

# Guardar predicciones en el dataset
df_operativas['dias_restantes_predichos'] = best_reg_model.predict(X)

# ============================================================================
# 3. MODELO 2: CLASIFICACIÓN - RIESGO DE FALLA
# ============================================================================

print("\n[3/6] Modelo 2: Clasificación - Riesgo de Falla (30, 60, 90 días)...")

# Crear variable target: riesgo de falla en próximos 30, 60, 90 días
def clasificar_riesgo_falla(dias_restantes):
    if dias_restantes <= 30:
        return 'ALTO'  # Falla en 30 días
    elif dias_restantes <= 90:
        return 'MEDIO'  # Falla en 60-90 días
    else:
        return 'BAJO'  # Más de 90 días

df_operativas['riesgo_falla'] = df_operativas['dias_restantes_estimados'].apply(clasificar_riesgo_falla)

print(f"   Distribución de clases:")
print(df_operativas['riesgo_falla'].value_counts().to_string())

# Preparar datos
y_clasificacion = df_operativas['riesgo_falla']
le_riesgo = LabelEncoder()
y_clasificacion_encoded = le_riesgo.fit_transform(y_clasificacion)

# Split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clasificacion_encoded, test_size=0.2, random_state=42, stratify=y_clasificacion_encoded
)

# Random Forest Classifier
print("\n   Entrenando Random Forest Classifier...")
rf_clf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_clf_model.fit(X_train_clf, y_train_clf)
y_pred_clf = rf_clf_model.predict(X_test_clf)
y_pred_proba = rf_clf_model.predict_proba(X_test_clf)

# Métricas
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"\n   Accuracy: {accuracy:.4f}")

print(f"\n   Classification Report:")
print(classification_report(y_test_clf, y_pred_clf, 
                          target_names=le_riesgo.classes_,
                          digits=3))

# Matriz de confusión
cm = confusion_matrix(y_test_clf, y_pred_clf)
print(f"\n   Matriz de Confusión:")
print(f"   {cm}")

# Guardar probabilidades
df_operativas['prob_falla_alto'] = rf_clf_model.predict_proba(X)[:, np.where(le_riesgo.classes_ == 'ALTO')[0][0]]
df_operativas['prob_falla_medio'] = rf_clf_model.predict_proba(X)[:, np.where(le_riesgo.classes_ == 'MEDIO')[0][0]]
df_operativas['prob_falla_bajo'] = rf_clf_model.predict_proba(X)[:, np.where(le_riesgo.classes_ == 'BAJO')[0][0]]

# ============================================================================
# 4. MODELO 3: ANÁLISIS DE SUPERVIVENCIA
# ============================================================================

print("\n[4/6] Modelo 3: Análisis de Supervivencia...")

# Preparar datos para análisis de supervivencia
# Usar TODAS las luminarias (operativas + falladas)
df_survival = df.copy()

# Duración: días desde instalación hasta falla o censura
df_survival['duracion_dias'] = (pd.to_datetime('2026-02-11') - df_survival['fecha_instalacion']).dt.days

# Evento: 1 si falló, 0 si está censurado (operativo)
df_survival['evento'] = (df_survival['estado_actual'] == 'REEMPLAZADO').astype(int)

# Para las que fallaron, usar fecha real de falla
mask_falla = df_survival['estado_actual'] == 'REEMPLAZADO'
df_survival.loc[mask_falla, 'duracion_dias'] = (
    df_survival.loc[mask_falla, 'fecha_falla'] - 
    df_survival.loc[mask_falla, 'fecha_instalacion']
).dt.days

print(f"   • Total observaciones: {len(df_survival):,}")
print(f"   • Eventos (fallas): {df_survival['evento'].sum():,}")
print(f"   • Censurados (operativos): {(1-df_survival['evento']).sum():,}")

# 4a) Kaplan-Meier: Curvas de supervivencia por tipo de LED
print("\n   a) Análisis Kaplan-Meier por tipo de LED...")

kmf = KaplanMeierFitter()

fig, ax = plt.subplots(figsize=(14, 8))

for tipo in df_survival['tipo_led'].unique():
    mask = df_survival['tipo_led'] == tipo
    kmf.fit(
        durations=df_survival.loc[mask, 'duracion_dias'],
        event_observed=df_survival.loc[mask, 'evento'],
        label=tipo
    )
    kmf.plot_survival_function(ax=ax, ci_show=False)

ax.set_xlabel('Días desde Instalación', fontsize=12)
ax.set_ylabel('Probabilidad de Supervivencia', fontsize=12)
ax.set_title('Curvas de Supervivencia por Tipo de LED\n(Análisis Kaplan-Meier)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/03_curvas_supervivencia_km.png', dpi=300, bbox_inches='tight')
print(f"      ✓ Gráfico guardado: 03_curvas_supervivencia_km.png")

# Tiempo medio de supervivencia por tipo
print(f"\n   Vida útil mediana por tipo de LED:")
for tipo in df_survival['tipo_led'].unique():
    mask = df_survival['tipo_led'] == tipo
    kmf.fit(
        durations=df_survival.loc[mask, 'duracion_dias'],
        event_observed=df_survival.loc[mask, 'evento']
    )
    mediana = kmf.median_survival_time_
    print(f"      {tipo}: {mediana:.0f} días ({mediana/365:.1f} años)")

# 4b) Cox Proportional Hazards Model
print("\n   b) Modelo de Cox (Proportional Hazards)...")

# Preparar datos para Cox
df_cox = df_survival[['duracion_dias', 'evento', 'horas_operacion_dia', 
                      'temperatura_promedio_c', 'humedad_promedio_pct',
                      'ciclos_encendido_dia', 'potencia_watts']].copy()

# Encodear tipo y marca
df_cox['tipo_led'] = le_tipo.fit_transform(df_survival['tipo_led'])
df_cox['marca'] = le_marca.fit_transform(df_survival['marca'])
df_cox['criticidad'] = le_criticidad.fit_transform(df_survival['criticidad'])

# Eliminar infinitos y NaN
df_cox = df_cox.replace([np.inf, -np.inf], np.nan).dropna()

print(f"      • Observaciones válidas: {len(df_cox):,}")

# Entrenar modelo Cox
cph = CoxPHFitter()
cph.fit(df_cox, duration_col='duracion_dias', event_col='evento')

print(f"\n   Resumen del Modelo de Cox:")
print(cph.summary[['coef', 'exp(coef)', 'p']].round(4).to_string())

# Concordance Index (C-index)
print(f"\n   Concordance Index: {cph.concordance_index_:.4f}")
print(f"   (0.5 = aleatorio, 1.0 = perfecto)")

# ============================================================================
# 5. VISUALIZACIONES DE MODELOS
# ============================================================================

print("\n[5/6] Generando visualizaciones de modelos...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Evaluación de Modelos Predictivos\nMantenimiento Iluminación LED - Clínica Alemana',
             fontsize=16, fontweight='bold')

# 1. Predicciones vs Real (Regresión)
axes[0, 0].scatter(y_test_reg, y_pred_rf, alpha=0.5, s=20)
axes[0, 0].plot([y_test_reg.min(), y_test_reg.max()], 
                [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Días Restantes Reales')
axes[0, 0].set_ylabel('Días Restantes Predichos')
axes[0, 0].set_title(f'Predicción vs Real\n(R² = {r2_rf:.4f})')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuos
residuos = y_test_reg - y_pred_rf
axes[0, 1].scatter(y_pred_rf, residuos, alpha=0.5, s=20)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Días Restantes Predichos')
axes[0, 1].set_ylabel('Residuos')
axes[0, 1].set_title('Análisis de Residuos')
axes[0, 1].grid(True, alpha=0.3)

# 3. Importancia de características (Regresión)
top_features = feature_importance_reg.head(10)
axes[0, 2].barh(range(len(top_features)), top_features['importance'].values)
axes[0, 2].set_yticks(range(len(top_features)))
axes[0, 2].set_yticklabels(top_features['feature'].values, fontsize=9)
axes[0, 2].set_xlabel('Importancia')
axes[0, 2].set_title('Top 10 Variables Importantes\n(Random Forest)')
axes[0, 2].grid(True, alpha=0.3, axis='x')

# 4. Matriz de Confusión (Clasificación)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=le_riesgo.classes_,
            yticklabels=le_riesgo.classes_,
            cbar_kws={'label': 'Cantidad'})
axes[1, 0].set_xlabel('Predicho')
axes[1, 0].set_ylabel('Real')
axes[1, 0].set_title(f'Matriz de Confusión\n(Accuracy = {accuracy:.3f})')

# 5. Distribución de probabilidades por clase real
for i, clase in enumerate(le_riesgo.classes_):
    mask = y_test_clf == i
    if mask.sum() > 0:
        axes[1, 1].hist(y_pred_proba[mask, i], bins=20, alpha=0.5, 
                       label=f'{clase} (real)', edgecolor='black')
axes[1, 1].set_xlabel('Probabilidad Predicha')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].set_title('Distribución de Probabilidades')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Importancia de características (Clasificación)
feature_importance_clf = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_clf_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

axes[1, 2].barh(range(len(feature_importance_clf)), feature_importance_clf['importance'].values)
axes[1, 2].set_yticks(range(len(feature_importance_clf)))
axes[1, 2].set_yticklabels(feature_importance_clf['feature'].values, fontsize=9)
axes[1, 2].set_xlabel('Importancia')
axes[1, 2].set_title('Top 10 Variables Importantes\n(Clasificación)')
axes[1, 2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/04_evaluacion_modelos.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Gráfico guardado: 04_evaluacion_modelos.png")

# ============================================================================
# 6. GUARDAR RESULTADOS Y MODELOS
# ============================================================================

print("\n[6/6] Guardando resultados...")

# Guardar dataset con predicciones
df_operativas_export = df_operativas[[
    'id_luminaria', 'piso', 'area', 'tipo_led', 'marca', 'criticidad',
    'horas_uso_acumuladas', 'porcentaje_vida_consumida',
    'dias_restantes_estimados', 'dias_restantes_predichos',
    'nivel_riesgo', 'prioridad_mantenimiento',
    'prob_falla_alto', 'prob_falla_medio', 'prob_falla_bajo'
]]

df_operativas_export.to_csv('/home/predicciones_mantenimiento.csv', 
                            index=False, encoding='utf-8-sig')
print(f"   ✓ Archivo guardado: predicciones_mantenimiento.csv")

# Resumen de métricas
metricas_resumen = {
    'Modelo': ['Regresión Lineal', 'Random Forest Reg', 'Gradient Boosting', 
               'Random Forest Clf', 'Cox PH'],
    'Tipo': ['Regresión', 'Regresión', 'Regresión', 'Clasificación', 'Supervivencia'],
    'Métrica_Principal': ['R²', 'R²', 'R²', 'Accuracy', 'C-Index'],
    'Valor': [r2_lr, r2_rf, r2_gb, accuracy, cph.concordance_index_]
}

df_metricas = pd.DataFrame(metricas_resumen)
df_metricas.to_csv('/home/metricas_modelos.csv', index=False)
print(f"   ✓ Archivo guardado: metricas_modelos.csv")

# ============================================================================
# 7. RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMEN DE MODELOS PREDICTIVOS")
print("="*80)

print("\n1. MODELO DE REGRESIÓN (Predicción de Días Restantes):")
print(f"   • Mejor modelo: {best_reg_name}")
print(f"   • R²: {max(r2_rf, r2_gb):.4f}")
print(f"   • MAE: {min(mae_rf, mae_gb):.2f} días")
print(f"   • RMSE: {min(rmse_rf, rmse_gb):.2f} días")
print(f"   • Interpretación: El modelo explica {max(r2_rf, r2_gb)*100:.1f}% de la variabilidad")

print("\n2. MODELO DE CLASIFICACIÓN (Riesgo de Falla):")
print(f"   • Accuracy: {accuracy:.4f}")
print(f"   • Clases: {', '.join(le_riesgo.classes_)}")
print(f"   • Uso: Identificar luminarias en riesgo alto/medio/bajo")

print("\n3. ANÁLISIS DE SUPERVIVENCIA:")
print(f"   • Concordance Index: {cph.concordance_index_:.4f}")
print(f"   • Curvas K-M por tipo de LED generadas")
print(f"   • Factores de riesgo identificados en modelo Cox")

print("\n4. VARIABLES MÁS IMPORTANTES:")
print(f"   • {feature_importance_reg.iloc[0]['feature']}")
print(f"   • {feature_importance_reg.iloc[1]['feature']}")
print(f"   • {feature_importance_reg.iloc[2]['feature']}")

print("\n" + "="*80)
print("\n✅ MODELOS PREDICTIVOS COMPLETADOS\n")
print("Archivos generados:")
print("  • predicciones_mantenimiento.csv")
print("  • metricas_modelos.csv")
print("  • 03_curvas_supervivencia_km.png")
print("  • 04_evaluacion_modelos.png")
print("="*80)
