### Solution path for a toy example ###
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../..')
from modules import full_path, path_solver

# Setting
X=np.array([[2,1,0],[1,2,1]], dtype=float)
Lambda = np.array([6,4,2], dtype=float)
y = np.array([15,5], dtype=float)

# Numba compilation
_ = full_path(X, y , Lambda, ratio=0., k_max=0., rtol_pattern=1e-6, atol_pattern = 1e-6, rtol_gamma=1e-6, split_max=1e1, log=0)
_ = path_solver(X, y, Lambda, k_max=0., rtol_pattern=1e-6, atol_pattern = 1e-6, rtol_gamma=1e-6, split_max=1e1, log=0)

# Full path
Gamma, Sol, Primal, Gap, M, Split, T = full_path(X, y , Lambda, ratio=1., k_max=1e3, rtol_pattern=1e-8, atol_pattern = 1e-10, rtol_gamma=1e-10, split_max=1e1, log=1)

# Solution path
fig, ax = plt.subplots()
ax.plot([1.1*Gamma[0]] + Gamma, [Sol[0]] + Sol)
for gamma in Gamma:
    ax.axvline(gamma, color='k', linestyle=':')
ax.axhline(0, color='k', linestyle=':', xmax=0.95)
ax.set_title('Solution path')
plt.show() 

# Full path with ridge regularization
eps = np.sqrt(2e-4)
Gamma2, Sol2, Primal2, Gap2, M2, Split2, T2 = full_path(np.concatenate((X, eps*np.eye(X.shape[-1]))), np.concatenate((y, np.zeros(X.shape[-1]))) , Lambda, ratio=1., k_max=1e3, rtol_pattern=1e-8, atol_pattern = 1e-10, rtol_gamma=1e-10, split_max=1e1, log=1)
print(f'\nwith no ridge regularization: nodes = {np.array(Gamma)}\n')
print(f'with ridge regularization: nodes = {np.array(Gamma2)}\n')


# Solver 
gamma = 3.
sol, (primal, gap), k = path_solver(X, y , gamma*Lambda, k_max=1e3, rtol_pattern=1e-8, atol_pattern = 1e-10, rtol_gamma=1e-10, split_max=1e1, log=0)
print(f'solution for gamma = {gamma}: {sol}')