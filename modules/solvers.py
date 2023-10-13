import numpy as np
from timeit import default_timer as timer
from numba import njit, objmode
from modules.utils import sorted_L1_norm, dual_sorted_L1_norm, PD_gap, pattern, face_pattern, pattern_matrix, affine_components, affine_components_LASSO

########### SLOPE ############

@njit
def gamma_fuse(a_s, b_s, gamma, gamma_min, tol):
    a = np.append(a_s, 0)
    b = np.append(b_s, 0)
    Gamma = - (b[1:]- b[:-1]) / (a[1:] - a[:-1])
    # Gamma = Gamma[(gamma_min <= Gamma) & (Gamma <= gamma - tol)]
    K = (gamma_min <= Gamma) & (Gamma <= gamma - tol)
    return max(Gamma[K]) if K.any() else gamma_min

@njit
def gamma_split(a_g, b_g, gamma, Lambda, rtol, atol, count_max):
    K = np.array([True]); count = 0
    L_cum = np.cumsum(Lambda)
    while K.any() and count < count_max:
        count += 1
        g = a_g * gamma + b_g
        sign = np.sign(g)
        perm = np.argsort(-np.abs(g))
        A_cum = np.cumsum(sign[perm]*a_g[perm])
        B_cum = np.cumsum(sign[perm]*b_g[perm])
        K = A_cum * gamma + B_cum - L_cum * gamma > atol + rtol*gamma*L_cum
        gamma = max(B_cum[K] / (L_cum[K] - A_cum[K])) if K.any() else gamma
    return gamma, count

@njit
def full_path(X, y, Lambda, ratio, k_max, rtol_pattern, atol_pattern, rtol_gamma, split_max, log):
    with objmode(t_in='float64'):
        t_in = timer()
    k = 0
    sol = np.zeros(X.shape[-1]) # m = pattern(sol, rtol_pattern); _ = pattern_matrix(m)  # for Numba compilation (not necessary)
    Sol = [sol]; Primal = [0]; Gap = [0]
    grad = X.T @ y; gamma = dual_sorted_L1_norm(grad, Lambda); Gamma = [gamma]
    m = face_pattern(grad, Lambda, rtol_pattern, atol_pattern); M = [m]
    count = 2; Split = [count>1]
    tol_gamma = rtol_gamma * gamma
    gamma_min = (1-ratio)*gamma
    with objmode(t_out='float64'):
        t_out = timer()
    T = [t_out - t_in]
    if log:
        with objmode():
            print('node {}: gamma = {:.3f}, elapsed time = {:.2e}s, gap = {:.2e}'.format(k, gamma, t_out - t_in, 0))
    while gamma > gamma_min and k < k_max:
        k += 1
        a_s,b_s,a_g,b_g = affine_components(X, y, Lambda, m)
        gamma_f = gamma_fuse(a_s, b_s, gamma, gamma_min, tol_gamma)
        gamma_s, count = gamma_split(a_g, b_g, gamma_f, Lambda, rtol_pattern, atol_pattern, split_max)
        if gamma_s > gamma - tol_gamma:
            print("Stopping: nodes too closed or wrong pattern (gamma_s > gamma - tol)")
            break
        elif count >= split_max:
            print("Stopping: not able to compute gamma_split (split_max reached)")
            break
        if count>1:
            gamma = gamma_s
            sol = pattern_matrix(m) @ (a_s * gamma + b_s) # solution in gamma
            grad = a_g * gamma + b_g
            m = face_pattern(grad, Lambda, rtol_pattern, atol_pattern) # pattern on the left of gamma
        else:
            gamma = gamma_f
            sol = pattern_matrix(m) @ (a_s * gamma + b_s) # solution in gamma
            m = pattern(sol, rtol_pattern)  # pattern on the left of gamma
        primal, gap = PD_gap(sol, X, y, gamma*Lambda)
        Gamma.append(gamma); Sol.append(sol) # have been checked
        Primal.append(primal); Gap.append(gap)
        M.append(m); Split.append(count>1) # have not been checked, do not return the last value
        with objmode(t_out='float64'):
            t_out = timer()
        T.append(t_out - t_in)
        if log:
            with objmode():
                print('node {}: gamma = {:.3f}, elapsed time = {:.2e}s, gap = {:.2e}'.format(k, gamma, t_out - t_in, gap)) 
    return Gamma, Sol, Primal, Gap, M[:-1], Split[:-1], T

@njit
def path_solver(X, y, Lambda, k_max, rtol_pattern, atol_pattern, rtol_gamma, split_max, log):
    k = 0; count = 2 # to enter the while loop
    sol = np.zeros(X.shape[-1]); # m = pattern(sol, rtol_pattern); _ = pattern_matrix(m)  # for Numba compilation
    grad = X.T @ y; run_gamma = dual_sorted_L1_norm(grad, Lambda)
    m = face_pattern(grad, Lambda, rtol_pattern, atol_pattern)
    tol_gamma = rtol_gamma * run_gamma
    if log:
        with objmode():
            print('node {}: running gamma = {:.3f}'.format(k, run_gamma))
    while run_gamma > 1. and k < k_max:
        k += 1
        a_s,b_s,a_g,b_g = affine_components(X, y, Lambda, m)
        gamma_f = gamma_fuse(a_s, b_s, run_gamma, 1., tol_gamma)
        gamma_s, count = gamma_split(a_g, b_g, gamma_f, Lambda, rtol_pattern, atol_pattern, split_max)
        if gamma_s > run_gamma - tol_gamma:
            print("Stopping: nodes too closed or wrong pattern (gamma_s > run_gamma - tol)")
            break
        elif count >= split_max:
            print("Stopping: not able to compute gamma_split (split_max reached)")
            break
        if count>1:
            run_gamma = gamma_s
            sol = pattern_matrix(m) @ (a_s * run_gamma + b_s) # solution in gamma
            grad = a_g * run_gamma + b_g
            m = face_pattern(grad, Lambda, rtol_pattern, atol_pattern) # pattern on the left of gamma
        else:
            run_gamma = gamma_f
            sol = pattern_matrix(m) @ (a_s * run_gamma + b_s) # solution in gamma
            m = pattern(sol, rtol_pattern)  # pattern on the left of gamma
        if log:
            with objmode():
                print('node {}: running gamma = {:.3f}'.format(k, run_gamma))
    return sol, PD_gap(sol, X, y, Lambda), k


############### LASSO #################

@njit
def gamma_zero(a_s, b_s, gamma, gamma_min, tol):
    Gamma = -b_s/a_s
    K = (gamma_min <= Gamma) & (Gamma <= gamma - tol)
    return max(Gamma[K]) if K.any() else gamma_min

@njit
def gamma_gradient(a_g, b_g, gamma, rtol, atol, count_max):
    K = np.array([True]); count = 0
    while K.any() and count < count_max:
        count +=1
        g = a_g * gamma + b_g 
        sign = np.sign(g)
        K = np.abs(g) - gamma > atol + rtol*gamma
        gamma = max(b_g[K] / (sign[K] - a_g[K])) if K.any() else gamma
    return gamma, count


@njit
def full_path_LASSO(X, y, ratio, k_max, rtol_sign, atol_sign, rtol_gamma, split_max, log):
    with objmode(t_in='float64'):
        t_in = timer()
    k = 0
    sol = np.zeros(X.shape[-1])
    Sol = [sol] #; Primal = [0]; Gap = [0]
    grad = X.T @ y; gamma = np.linalg.norm(grad,ord=np.inf); Gamma = [gamma]
    s = np.zeros(X.shape[-1]); s[np.isclose(grad, gamma, rtol_sign, atol_sign)] = 1; s[np.isclose(grad, -gamma, rtol_sign, atol_sign)] = -1; S = [s]
    count = 2; Split = [count>1]
    tol_gamma = rtol_gamma * gamma
    gamma_min = (1-ratio)*gamma
    with objmode(t_out='float64'):
        t_out = timer()
    T = [t_out - t_in]
    if log:
        with objmode():
            print('node {}: gamma = {:.3f}, elapsed time = {:.2e}s'.format(k, gamma, t_out - t_in)) #, gap = {:.2e} ,0
    while gamma > gamma_min and k < k_max:
        k += 1
        a_s,b_s,a_g,b_g = affine_components_LASSO(X, y, s)
        gamma_z = gamma_zero(a_s, b_s, gamma, gamma_min, tol_gamma)
        gamma_g, count = gamma_gradient(a_g, b_g, gamma_z, rtol_sign, atol_sign, split_max)
        if gamma_g > gamma - tol_gamma:
            print("Stopping: nodes too closed or wrong sign (gamma_g > gamma - tol)")
            break
        elif count >= split_max:
            print("Stopping: not able to compute gamma_gradient (split_max reached)")
            break
        if count>1:
            gamma = gamma_g
            sol = np.zeros(X.shape[-1]); sol[s!=0] = a_s * gamma + b_s # solution in gamma
            grad = a_g * gamma + b_g
            s = np.zeros(X.shape[-1]); s[np.isclose(grad, gamma, rtol_sign, atol_sign)] = 1; s[np.isclose(grad, -gamma, rtol_sign, atol_sign)] = -1  # sign on the left of gamma
        else:
            gamma = gamma_z
            sol = np.zeros(X.shape[-1]); sol[s!=0] = a_s * gamma + b_s # solution in gamma
            sol[np.abs(sol) < atol_sign] = 0
            s = np.sign(sol)
        Gamma.append(gamma); Sol.append(sol) # have been checked
        # Primal.append(primal); Gap.append(gap)
        S.append(s); Split.append(count>1) # have not been checked, do not return the last value
        with objmode(t_out='float64'):
            t_out = timer()
        T.append(t_out - t_in)
        if log:
            with objmode():
                print('node {}: gamma = {:.3f}, elapsed time = {:.2e}s'.format(k, gamma, t_out - t_in)) # , gap = {:.2e} , gap
    return Gamma, Sol, S[:-1], Split[:-1], T #, Primal, Gap