### Solution path for the `Riboflavin` data set ###

# The data set can be downloaded from <https://www.annualreviews.org/doi/suppl/10.1146/annurev-statistics-022513-115545>.

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

sys.path.append('../..')
from modules import full_path, path_solver, dual_sorted_L1_norm as dual_norm, pattern

# Import of the data set
ribo = pd.read_csv('../datasets/riboflavin.csv', sep=',', index_col=0).T
data = ribo.drop(columns='q_RIBFLV')
target = ribo['q_RIBFLV']

# Setting (and data standardization)
X = data.to_numpy(dtype=float)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)
X = (X - X_mean) / X_std

y = target.to_numpy(dtype=float)
y_mean = y.mean()
y = y - y_mean 

Lambda = np.linspace(4,1,X.shape[-1],dtype=float)

# Numba compilation
_ = full_path(X, y , Lambda, ratio=0., k_max=0., rtol_pattern=1e-6, atol_pattern = 1e-6, rtol_gamma=1e-6, split_max=1e1, log=0)
_ = path_solver(X, y, Lambda, k_max=0., rtol_pattern=1e-6, atol_pattern = 1e-6, rtol_gamma=1e-6, split_max=1e1, log=0)

# (Almost) full path (ratio of it)
Gamma, Sol, Primal, Gap, M, Split, T = full_path(X, y , Lambda, ratio=0.98, k_max=1e4, rtol_pattern=1e-10, atol_pattern = 1e-10, rtol_gamma=1e-10, split_max=1e1, log=1)

# Solution path
K = 100
fig, ax = plt.subplots()
ax.plot([1.1*Gamma[0]] + Gamma[:K], [Sol[0]] + Sol[:K])
# for gamma in Gamma[:K]:
#     ax.axvline(gamma, color='k', linestyle=':')
ax.axhline(0, color='k', linestyle=':', xmax=0.95)
# ax.set_xscale('symlog')
ax.set_title(f'Solution path (first {K} nodes)')
plt.show() 

# Solver 
frac = 0.5; gamma = frac*dual_norm(X.T@y, Lambda)
t_start = timer()
sol, (primal, gap), k = path_solver(X, y , gamma*Lambda, k_max=1e3, rtol_pattern=1e-10, atol_pattern = 1e-10, rtol_gamma=1e-10, split_max=1e1, log=0)
print(f'pattern for {frac} x gamma_max: {pattern(sol, tol=1e-10)}')
print(f'elapsed time = {timer() - t_start:.2e}s, crossed nodes = {k}, primal-dual gap = {gap:.2e}')