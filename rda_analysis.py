import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skbio.stats.ordination import rda
from scipy.stats import gmean

# ============================================================
# 1. CARGA Y TRANSFORMACIÓN (Mantenemos tu lógica funcional)
# ============================================================
df_total = pd.read_csv("datos_rda_0.csv", sep="\t", index_col=0) 
df_total.columns = df_total.columns.str.strip()

cols_ambientales = ['TOC', 'TIC', 'TN', 'Po', 'Pi', 'TOC_TN', 'COT_Po', 'pH', 'TN_Po']
cols_generos = [c for c in df_total.columns if c not in cols_ambientales]

df_bio = df_total[cols_generos]
df_env = df_total[cols_ambientales]

def clr_transform(X):
    X_pseudo = X + 1 
    gm = gmean(X_pseudo, axis=1)
    return np.log(X_pseudo.divide(gm, axis=0))

df_bio_clr = clr_transform(df_bio)
df_env_std = (df_env - df_env.mean()) / df_env.std()

# ============================================================
# 2. ANÁLISIS RDA
# ============================================================
rda_output = rda(df_bio_clr, df_env_std, scale_Y=False)

# ============================================================
# 3. GRÁFICO TRIPLOT CON LEYENDA DE TAXAS
# ============================================================
# Aumentamos el ancho para que quepa la leyenda a la derecha
fig, ax = plt.subplots(figsize=(14, 10)) 

samples = rda_output.samples.iloc[:, :2]     
features = rda_output.features.iloc[:, :2]   
biplot = rda_output.biplot_scores.iloc[:, :2] 

# --- A. CAPAS (Puntos de color) ---
scatter = ax.scatter(samples.iloc[:, 0], samples.iloc[:, 1], 
                    c=range(len(samples)), cmap='coolwarm', s=150, edgecolors='k', zorder=5)

for i, txt in enumerate(samples.index):
    ax.annotate(txt, (samples.iloc[i, 0], samples.iloc[i, 1]), xytext=(7,7), 
                textcoords='offset points', fontweight='bold', fontsize=10)

# --- B. GÉNEROS (Círculos numerados + Panel derecho) ---
top_taxa = features.abs().sum(axis=1).sort_values(ascending=False).head(12).index
genero_legend_elements = []

for idx, taxon in enumerate(top_taxa, 1):
    x_feat = features.loc[taxon].iloc[0]
    y_feat = features.loc[taxon].iloc[1]
    
    # Dibujar pequeño círculo para el género
    ax.plot(x_feat, y_feat, 'o', color='darkgreen', markersize=18, alpha=0.6, zorder=4)
    # Poner el número dentro
    ax.text(x_feat, y_feat, str(idx), color='white', fontsize=9, 
            fontweight='bold', ha='center', va='center', zorder=4)
    
    # Guardar para la leyenda (Número: Nombre del Género)
    genero_legend_elements.append(f"{idx}: {taxon}")

# --- C. VECTORES AMBIENTALES (Flechas Rojas) ---
for i, var in enumerate(biplot.index):
    x_vec, y_vec = biplot.iloc[i, 0], biplot.iloc[i, 1]
    ax.arrow(0, 0, x_vec, y_vec, color='red', head_width=0.03, alpha=0.8, width=0.005, zorder=3)
    ax.text(x_vec * 1.15, y_vec * 1.15, var, color='red', fontweight='bold', 
            fontsize=11, ha='center', va='center')

# --- AJUSTES FINALES Y LEYENDA ---

# Crear el panel de taxas al lado derecho
legend_text = "\n".join(genero_legend_elements)
ax.text(1.05, 0.95, "Key Taxa (Top 12):", transform=ax.transAxes, 
        fontsize=12, fontweight='bold', va='top', color='darkgreen')
ax.text(1.05, 0.90, legend_text, transform=ax.transAxes, 
        fontsize=10, va='top', linespacing=1.8, style='italic')

# Encuadre dinámico
all_x = np.concatenate([samples.iloc[:,0], features.loc[top_taxa].iloc[:,0], biplot.iloc[:,0]])
all_y = np.concatenate([samples.iloc[:,1], features.loc[top_taxa].iloc[:,1], biplot.iloc[:,1]])
margin = 0.2
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

# Barra de color inferior
cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.15, aspect=50, shrink=0.7)
cbar.set_label('Gradiente de Capas (Superficie L1 $\\rightarrow$ Fondo L11)', fontsize=11)

# Estética de ejes
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.axvline(0, color='black', lw=0.8, ls='--')
expl1 = rda_output.proportion_explained.iloc[0] * 100
expl2 = rda_output.proportion_explained.iloc[1] * 100
ax.set_xlabel(f"RDA1 ({expl1:.2f}%)", fontsize=12)
ax.set_ylabel(f"RDA2 ({expl2:.2f}%)", fontsize=12)

plt.title("Análisis de Redundancia (RDA): Integración Generos-Fisicoquímica\n(Visualización por Claves Numéricas)", fontsize=14, pad=20)
plt.grid(alpha=0.2)

# El uso de 'rect' permite dejar espacio a la derecha para la leyenda
plt.tight_layout(rect=[0, 0, 0.85, 1]) 
plt.savefig("rda_final_numerado.png", dpi=300, bbox_inches='tight')
plt.show()