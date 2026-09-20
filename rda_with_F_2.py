import pandas as pd
import numpy as np
from scipy.stats import gmean

# ============================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================
df_total = pd.read_csv("datos_rda_1.csv", sep="\t", index_col=0) 
df_total.columns = df_total.columns.str.strip()

cols_ambientales = ['pH', 'Po', 'TN']
cols_generos = [c for c in df_total.columns if c not in ['TOC', 'TIC', 'TN', 'Po', 'Pi', 'TOC_TN', 'COT_Po', 'pH', 'TN_Po', 'C_N', 'C_Po', 'N_Po']]

df_bio = df_total[cols_generos]
df_env = df_total[cols_ambientales]

# Transformación CLR para la biología
def clr_transform(X):
    X_pseudo = X + 1 
    gm = gmean(X_pseudo, axis=1)
    return np.log(X_pseudo.divide(gm, axis=0))

Y = clr_transform(df_bio).values # Matriz biológica (n x p)
X = (df_env - df_env.mean()) / df_env.std() # Matriz ambiental estandarizada (n x q)

n, p = Y.shape
q = X.shape[1]

# Centrar Y
Y_centered = Y - Y.mean(axis=0)

# Función para calcular Suma de Cuadrados (SS) del RDA
def calc_rda_ss(Y_c, X_mat):
    H = X_mat.values @ np.linalg.inv(X_mat.T.values @ X_mat.values) @ X_mat.T.values
    Y_hat = H @ Y_c
    Y_res = Y_c - Y_hat
    return np.sum(Y_hat**2), np.sum(Y_res**2)

# ============================================================
# 2. MODELO GLOBAL (PSEUDO-F Y P-VALOR GLOBAL)
# ============================================================
ss_cond_real, ss_res_real = calc_rda_ss(Y_centered, X)
df_model = q
df_residual = n - q - 1

pseudo_F_real = (ss_cond_real / df_model) / (ss_res_real / df_residual)

n_permutaciones = 999
np.random.seed(42)

f_perm_global = []
for _ in range(n_permutaciones):
    idx = np.random.permutation(n)
    X_perm = X.iloc[idx, :]
    ss_c_p, ss_r_p = calc_rda_ss(Y_centered, X_perm)
    f_p = (ss_c_p / df_model) / (ss_r_p / df_residual)
    f_perm_global.append(f_p)

p_value_global = (np.sum(np.array(f_perm_global) >= pseudo_F_real) + 1) / (n_permutaciones + 1)

# ============================================================
# 3. EFECTOS MARGINALES (P-VALOR POR VARIABLE INDIVIDUAL)
# ============================================================
p_values_vars = {}

for var in cols_ambientales:
    # Modelo completo vs Modelo reducido (sin la variable de interés)
    X_reduced = X.drop(columns=[var])
    ss_cond_red, _ = calc_rda_ss(Y_centered, X_reduced)
    
    # Incremento de varianza explicado por la variable específica (SS_marginal)
    ss_var_real = ss_cond_real - ss_cond_red
    f_var_real = (ss_var_real / 1) / (ss_res_real / df_residual)
    
    f_perm_var = []
    for _ in range(n_permutaciones):
        # Permutación parcial de la variable específica
        X_perm_var = X.copy()
        X_perm_var[var] = np.random.permutation(X_perm_var[var])
        
        ss_c_p, ss_r_p = calc_rda_ss(Y_centered, X_perm_var)
        ss_var_p = ss_c_p - ss_cond_red
        f_p = (ss_var_p / 1) / (ss_r_p / df_residual)
        f_perm_var.append(f_p)
        
    p_val_var = (np.sum(np.array(f_perm_var) >= f_var_real) + 1) / (n_permutaciones + 1)
    p_values_vars[var] = p_val_var

# ============================================================
# 4. REPORTE DE RESULTADOS
# ============================================================
print("\n" + "="*50)
print(f"ESTADÍSTICO PSEUDO-F REAL (Modelo con 3 vars): {pseudo_F_real:.4f}")
print(f"VALOR DE P GLOBAL DEL MODELO                 : {p_value_global:.4f}")
print("="*50)
print("PRUEBAS MARGINALES PERMUTACIONALES POR VARIABLE:")
for var, p_val in p_values_vars.items():
    print(f"Variable: {var:<10} | p-value: {p_val:.4f}")
print("="*50 + "\n")