### Solution path for the `Wine Quality` data set ###

# The data set can be downloaded from <http://archive.ics.uci.edu/ml/datasets/Wine+Quality>.

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

sys.path.append('../..')
from modules import full_path, path_solver, dual_sorted_L1_norm as dual_norm, pattern, full_path_LASSO, min_SURE, min_SURE_LASSO

# Import of the data set
wine = pd.read_csv('../datasets/winequality-red.csv', sep=';')
data = wine.drop(columns=['quality'])
target = wine['quality']

# Setting (and data standardization)
X = data.to_numpy(dtype=float)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)
X = (X - X_mean) / X_std

y = target.to_numpy(dtype=float)
y_mean = y.mean()
y = y - y_mean

# Lambda = np.linspace(4,1,X.shape[-1],dtype=float)
Lambda = np.sqrt(range(1,12))-np.sqrt(range(0,11))

# Numba compilation
_ = full_path(X, y , Lambda, ratio=0., k_max=0., rtol_pattern=1e-6, atol_pattern = 1e-6, rtol_gamma=1e-6, split_max=1e1, log=0)
_ = path_solver(X, y, Lambda, k_max=0., rtol_pattern=1e-6, atol_pattern = 1e-6, rtol_gamma=1e-6, split_max=1e1, log=0)
_ = full_path_LASSO(X, y, ratio=0., k_max=0., rtol_sign=1e-6, atol_sign = 1e-6, rtol_gamma=1e-6, split_max=1e1, log=0)


#### SLOPE ######

# Full SLOPE path
Gamma, Sol, Primal, Gap, M, Split, T = full_path(X, y , Lambda, ratio=1., k_max=1e4, rtol_pattern=1e-10, atol_pattern = 1e-10, rtol_gamma=1e-10, split_max=1e1, log=1)

# Absolute SLOPE solution path + ols
abs_Sol = [np.abs(sol) for sol in Sol]

ols = np.linalg.solve(X.T@X, X.T@y)
abs_ols = np.abs(ols)
    
fig, ax = plt.subplots()
ax.plot([1.1*Gamma[0]] + Gamma, [Sol[0]] + abs_Sol)
for gamma in Gamma:
    ax.axvline(gamma, color='k', linestyle=':')
ax.axhline(0, color='k', linestyle=':', xmax=0.95)
ax.plot(0, [np.abs(ols)], 'rx')
# ax.set_xscale('symlog')
ax.set_title('Absolute SLOPE solution path')
plt.show() 

fig.savefig('../results/wine-quality_SLOPEpath', bbox_inches="tight", pad_inches=0.05)

# SLOPE solver 
frac = 0.5; gamma = frac*dual_norm(X.T@y, Lambda)
t_start = timer()
sol, (primal, gap), k = path_solver(X, y , gamma*Lambda, k_max=1e3, rtol_pattern=1e-10, atol_pattern = 1e-10, rtol_gamma=1e-10, split_max=1e1, log=0)
print(f'pattern for {frac} x gamma_max: {pattern(sol, tol=1e-10)}')
print(f'elapsed time = {timer() - t_start:.2e}s, crossed nodes = {k}, primal-dual gap = {gap:.2e}')

# SURE exact minimization for SLOPE
sigma2 = np.linalg.norm(y-X@ols,ord=2)**2/(X.shape[0]-X.shape[1])
Critical, SURE, Sol = min_SURE(X, y, Lambda, Gamma, M, sigma2, tol=1e-10)   
i0 = np.argmin(SURE)
print(f'SURE minimizer: gamma = {Critical[i0]:.3f}')
print(f'SURE minimum: sure(gamma) = {SURE[i0]:.3f}')     
print(f'solution in gamma: {Sol[i0]}')  
print(f'solution pattern in gamma: {pattern(Sol[i0],tol=1e-10)}')


########### LASSO ##############

# Full LASSO path
Gamma, Sol, S, Split, T = full_path_LASSO(X, y, ratio=1., k_max=1e4, rtol_sign=1e-10, atol_sign = 1e-10, rtol_gamma=1e-10, split_max=1e1, log=1)

# Absolute LASSO solution path + ols
abs_Sol = [np.abs(sol) for sol in Sol]

ols = np.linalg.solve(X.T@X, X.T@y)
abs_ols = np.abs(ols)
    
fig, ax = plt.subplots()
ax.plot([1.1*Gamma[0]] + Gamma, [Sol[0]] + abs_Sol)
for gamma in Gamma:
    ax.axvline(gamma, color='k', linestyle=':')
ax.axhline(0, color='k', linestyle=':', xmax=0.95)
ax.plot(0, [np.abs(ols)], 'rx')
# ax.set_xscale('symlog')
ax.set_title('Absolute LASSO solution path')
plt.show() 

fig.savefig('../results/wine-quality_LASSOpath', bbox_inches="tight", pad_inches=0.05)

# SURE exact minimization for LASSO