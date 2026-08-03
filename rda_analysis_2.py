import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skbio.stats.ordination import rda
from scipy.stats import gmean

# ============================================================
# 1. CARGA DE DATOS CON DETECCIÓN FLEXIBLE
# ============================================================
df_total = pd.read_csv("datos_rda_0.csv", sep="\t", index_col=0) 
if df_total.shape[1] <= 1:
    df_total = pd.read_csv("datos_rda_0.csv", sep=",", index_col=0)

df_total.columns = df_total.columns.str.strip()

# Identificar dinámicamente las variables geoquímicas
posibles_ambientales = ['TOC', 'TIC', 'TN', 'Po', 'Pi', 'C_N', 'C_Po', 'pH', 'N_Po', 'TOC_TN', 'COT_Po', 'TN_Po']
cols_ambientales = []

for pos in posibles_ambientales:
    encontrada = [c for c in df_total.columns if pos == c or pos in c]
    if encontrada and encontrada[0] not in cols_ambientales:
        cols_ambientales.append(encontrada[0])

if not cols_ambientales:
    cols_ambientales = [c for c in df_total.columns if any(p in c for p in posibles_ambientales)]

cols_generos = [c for c in df_total.columns if c not in cols_ambientales]

df_bio = df_total[cols_generos]
df_env = df_total[cols_ambientales]

# Transformación CLR para la biología
def clr_transform(X):
    X_pseudo = X + 1 
    gm = gmean(X_pseudo, axis=1)
    return np.log(X_pseudo.divide(gm, axis=0))

df_bio_clr = clr_transform(df_bio)

# Estandarización Z-score para la geoquímica
df_env_std = (df_env - df_env.mean()) / df_env.std()

# ============================================================
# 2. ANÁLISIS RDA Y ESTADÍSTICA DE PERMUTACIÓN (MONTE CARLO)
# ============================================================
rda_output = rda(df_bio_clr, df_env_std, scale_Y=False)

# A. Varianza Explicada por los ejes del gráfico (RDA1 y RDA2)
expl1 = rda_output.proportion_explained.iloc[0] * 100
expl2 = rda_output.proportion_explained.iloc[1] * 100
var_grafico = expl1 + expl2

# B. Permutación de Monte Carlo Global (999 Permutaciones)
n_permutaciones = 999
contador_r2_mayor = 0
np.random.seed(42)  # Reproducibilidad

for _ in range(n_permutaciones):
    env_permutado = df_env_std.sample(frac=1).reset_index(drop=True)
    env_permutado.index = df_env_std.index
    rda_perm = rda(df_bio_clr, env_permutado, scale_Y=False)
    
    # Compara la varianza explicada por los dos primeros ejes
    if rda_perm.proportion_explained.iloc[:2].sum() * 100 >= var_grafico:
        contador_r2_mayor += 1

p_val_global = (contador_r2_mayor + 1) / (n_permutaciones + 1)

# C. Pruebas Marginales Permutacionales por Variable (Tabla S1)
resultados_variables = {}
for var in cols_ambientales:
    contador_var = 0
    corr_real = df_bio_clr.corrwith(df_env_std[var]).abs().mean()
    
    for _ in range(n_permutaciones):
        var_permutada = np.random.permutation(df_env_std[var])
        corr_perm = df_bio_clr.corrwith(pd.Series(var_permutada, index=df_bio_clr.index)).abs().mean()
        if corr_perm >= corr_real:
            contador_var += 1
            
    resultados_variables[var] = (contador_var + 1) / (n_permutaciones + 1)

# Imprimir reporte formal y limpio en la consola
print("\n" + "="*60)
print("              REPORTE ESTADÍSTICO RDA (FULL MODEL)")
print("="*60)
print(f"Varianza Representada en Gráfico (RDA1 + RDA2): {var_grafico:.2f}%")
print(f"  -> RDA1: {expl1:.2f}% | RDA2: {expl2:.2f}%")
print(f"Valor de p Global (Monte Carlo 999 perm)      : {p_val_global:.4f}")
print("Nota: Modelo totalmente saturado (k >= n-2).")
print("-"*60)
print("PRUEBAS MARGINALES POR VARIABLE (Para Tabla Suplementaria S1):")
for var, p_val in resultados_variables.items():
    print(f" -> Variable: {var:<12} | p-value: {p_val:.4f}")
print("="*60 + "\n")

# ============================================================
# 3. MAPA DE REFERENCIA EXCLUSIVO (SOLO TOP 10 GÉNEROS)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 12))

eje_x_name = rda_output.samples.columns[0]
eje_y_name = rda_output.samples.columns[1]

samples = rda_output.samples[[eje_x_name, eje_y_name]]     
features = rda_output.features[[eje_x_name, eje_y_name]]   
biplot = rda_output.biplot_scores[[eje_x_name, eje_y_name]] 

# --- A. GÉNEROS SELECCIONADOS (SOLO EL TOP 10) ---
top_10_generos = (features[eje_x_name]**2 + features[eje_y_name]**2).sort_values(ascending=False).head(10).index

ax.scatter(features.loc[top_10_generos, eje_x_name], 
           features.loc[top_10_generos, eje_y_name], 
           color='darkgreen', s=40, zorder=3)

for gen in top_10_generos:
    x_gen = features.loc[gen, eje_x_name]
    y_gen = features.loc[gen, eje_y_name]
    gen_limpio = gen.replace('g__', '').replace('f__', '')
    ax.text(x_gen, y_gen, f"  {gen_limpio}", color='darkgreen', fontsize=10, 
            fontstyle='italic', zorder=4, fontweight='bold', va='center')

# --- B. CAPAS (Puntos con gradiente + Nombres) ---
scatter = ax.scatter(samples[eje_x_name], samples[eje_y_name], 
                    c=range(len(samples)), cmap='coolwarm', s=220, edgecolors='k', lw=1.5, zorder=5)

for i, txt in enumerate(samples.index):
    ax.annotate(txt, (samples.iloc[i, 0], samples.iloc[i, 1]), xytext=(9, 9), 
                textcoords='offset points', fontweight='bold', fontsize=12, color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', lw=0.5, pad=1.5))

# --- C. VECTORES AMBIENTALES (Flechas + Nombres) ---
for i, var in enumerate(biplot.index):
    x_vec = biplot.iloc[i, 0]
    y_vec = biplot.iloc[i, 1]
    ax.arrow(0, 0, x_vec, y_vec, color='red', head_width=0.03, alpha=0.8, width=0.005, zorder=4)
    ax.text(x_vec * 1.12, y_vec * 1.12, var, color='red', fontweight='bold', 
            fontsize=13, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# --- LÍMITES Y FUENTES ---
all_x = np.concatenate([samples[eje_x_name], features.loc[top_10_generos, eje_x_name], biplot[eje_x_name]])
all_y = np.concatenate([samples[eje_y_name], features.loc[top_10_generos, eje_y_name], biplot[eje_y_name]])
margin = 0.15
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

expl1 = rda_output.proportion_explained.iloc[0] * 100
expl2 = rda_output.proportion_explained.iloc[1] * 100
ax.set_xlabel(f"RDA1 ({expl1:.2f}%)", fontsize=16, fontweight='bold', labelpad=12)
ax.set_ylabel(f"RDA2 ({expl2:.2f}%)", fontsize=16, fontweight='bold', labelpad=12)

ax.tick_params(axis='both', which='major', labelsize=12)

cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.12, aspect=50, shrink=0.8)
cbar.set_label('Gradiente de Capas (Superficie L1 $\\rightarrow$ Fondo L11)', fontsize=14, labelpad=10)
cbar.ax.tick_params(labelsize=11)

plt.title("RDA: Mapa de Referencia Filtrado (Solo TOP 10)", fontsize=18, fontweight='bold', pad=25)

ax.axhline(0, color='black', lw=0.8, ls='--')
ax.axvline(0, color='black', lw=0.8, ls='--')
plt.grid(alpha=0.15)

plt.tight_layout()
plt.savefig("rda_mapa_referencia_filtrado.png", dpi=300)
print("Ejecutado con éxito. Gráfico 'rda_mapa_referencia_filtrado.png' generado.")
plt.show()
