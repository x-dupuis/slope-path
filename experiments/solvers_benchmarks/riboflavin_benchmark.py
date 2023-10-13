### Solvers benchmark on the `Riboflavin` data set ###

# The data set can be downloaded from <https://www.annualreviews.org/doi/suppl/10.1146/annurev-statistics-022513-115545>.

# The solvers (except our `path_solver`) were coded by Larsson et al. 
# for the benchmark of their paper *Coordinate Descent for SLOPE*; 
# their code is available at <https://github.com/jolars/slopecd>
# and needs to be installed to run the simulations here.

import sys
import numpy as np
import pandas as pd
from timeit import repeat

sys.path.append('../..')
from modules import path_solver, dual_sorted_L1_norm as dual_norm
from slope.solvers import prox_grad, admm, hybrid_cd #, newt_alm, oracle_cd

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

# # Numba compilation
# _ = path_solver(X, y, Lambda, k_max=0., rtol_pattern=1e-6, atol_pattern = 1e-6, rtol_gamma=1e-6, split_max=1e1, log=0)
# _ = hybrid_cd(X, y, Lambda, fit_intercept=False, tol=1e-1)
# # _ = oracle_cd(X, y, Lambda, fit_intercept=False, tol=1e-1)
# _ = admm(X, y, Lambda, fit_intercept=False, rho=100, adaptive_rho=False, tol=1e-1)
# # _ = newt_alm(X, y, Lambda, fit_intercept=False, tol=1e-1)
# _ = prox_grad(X, y, Lambda, fit_intercept=False, tol=1e-1)

# Check-up and numba compilation
gamma = dual_norm(X.T@y, Lambda) / 2
nb_loops = 1; nb_runs = 1

# FISTA
sol, intercept, primals, gaps, times = prox_grad(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, fista=True, tol=1e-12/X.shape[0], max_epochs=100_000)
time_fista =np.mean(repeat('prox_grad(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, fista=True, tol=1e-12/X.shape[0], max_epochs=100_000)',
                   repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
print(f'FISTA: {time_fista:.2e}s')
print(sol, X.shape[0]*primals[-1], X.shape[0]*gaps[-1], '\n')

# Anderson PGD
sol, intercept, primals, gaps, times = prox_grad(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, anderson=True, tol=1e-12/X.shape[0], max_epochs=100_000)
time_pgd =np.mean(repeat('prox_grad(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, anderson=True, tol=1e-12/X.shape[0], max_epochs=100_000)',
                   repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
print(f'Anderson PGD: {time_pgd:.2e}s')
print(sol, X.shape[0]*primals[-1], X.shape[0]*gaps[-1], '\n')

# ADMM (with rho=100)
sol, intercept, primals, gaps, times = admm(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, rho=100, adaptive_rho=False, tol=1e-12/X.shape[0], max_epochs=100_000)
time_admm =np.mean(repeat('admm(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, rho=100, adaptive_rho=False, tol=1e-12/X.shape[0], max_epochs=100_000)',
                   repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
print(f'ADMM: {time_admm:.2e}s')
print(sol, X.shape[0]*primals[-1], X.shape[0]*gaps[-1], '\n')

# hybrid CD
sol, intercept, primals, gaps, times, n_cluster = hybrid_cd(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, tol=1e-12/X.shape[0])
time_cd =np.mean(repeat('hybrid_cd(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, tol=1e-12/X.shape[0])',
                   repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
print(f'hybrid CD: {time_cd:.2e}s')
print(sol, X.shape[0]*primals[-1], X.shape[0]*gaps[-1], '\n')

# path (our)
sol, (primal, gap), k = path_solver(X, y, gamma*Lambda, k_max=1e3, rtol_pattern=1e-10, atol_pattern = 1e-10, rtol_gamma=1e-10, split_max=1e1, log=0)
time_path =np.mean(repeat('path_solver(X, y, gamma*Lambda, k_max=1e3, rtol_pattern=1e-10, atol_pattern = 1e-10, rtol_gamma=1e-10, split_max=1e1, log=0)',
                   repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
print(f'path (our): {time_path:.2e}s')
print(sol, primal, gap, '\n')

# # oracle CD
# sol, intercept, primals, gaps, times = oracle_cd(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, tol=1e-12)
# print(sol, X.shape[0]*primals[-1], gaps[-1])
# time_oracle =np.mean(repeat('oracle_cd(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, tol=1e-12)',
#                    repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
# print(f'oracle CD: {time_oracle:.2e}s')

# # Newt-ALM
# sol, intercept, primals, gaps, times = newt_alm(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, tol=1e-12)
# print(sol, X.shape[0]*primals[-1], gaps[-1])
# time_newt_alm =np.mean(repeat('newt_alm(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, tol=1e-12)',
#                    repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
# print(f'Newt-ALM: {time_newt_alm:.2e}s')

# Benchmark
nb_loops = 100; nb_runs = 7 

benchmark = pd.DataFrame({},index=['hybrid CD', 'path (our)']) #'FISTA', 'Anderson PGD', 'ADMM (rho=100)', 
benchmark.columns.name='gamma / gamma_max'

for ratio in [0.5, 0.1]:
    gamma = ratio*dual_norm(X.T@y, Lambda)
    # time_fista =np.mean(repeat('prox_grad(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, fista=True, tol=1e-12/X.shape[0], max_epochs=100_000)',
    #                     repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
    # time_pgd =np.mean(repeat('prox_grad(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, anderson=True, tol=1e-12/X.shape[0], max_epochs=100_000)',
    #                   repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
    # time_admm =np.mean(repeat('admm(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, rho=100, adaptive_rho=False, tol=1e-12/X.shape[0], max_epochs=100_000)',
    #                    repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
    time_cd =np.mean(repeat('hybrid_cd(X, y, gamma/X.shape[0]*Lambda, fit_intercept=False, tol=1e-12/X.shape[0])',
                     repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
    time_path =np.mean(repeat('path_solver(X, y, gamma*Lambda, k_max=1e3, rtol_pattern=1e-10, atol_pattern = 1e-10, rtol_gamma=1e-10, split_max=1e1, log=0)',
                       repeat=nb_runs, number=nb_loops, globals=globals()))/nb_loops
    benchmark[f'{ratio}'] = [time_admm, time_cd, time_path]
    
with pd.option_context('display.float_format', '{:,.2e}'.format):
    print(benchmark)

benchmark.to_csv('../results/riboflavin_benchmark.csv', float_format='{:.2e}'.format, mode='a')

# with open('../results/riboflavin_benchmark.txt', 'a') as f:
#                 f.write(benchmark.to_latex(float_format='{:.2e}'.format))


