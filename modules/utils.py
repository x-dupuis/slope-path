import numpy as np
from math import isnan, isinf
from numba import njit

@njit
def sorted_L1_norm(b,Lambda):
    b = - np.sort(-np.abs(b))  # x sorted by decreasing absolute value 
    return np.sum(Lambda*b)

@njit
def dual_sorted_L1_norm(v,Lambda):
    v = - np.sort(-np.abs(v)) # v sorted by decreasing absolute value 
    v_cum = np.cumsum(v)  # cumulative sum of v
    Lambda_cum = np.cumsum(Lambda) # cumulative sum of Lambda
    return np.max(v_cum/Lambda_cum)

@njit
def PD_gap(b, X, y, Lambda):
    res = y - X@b
    primal = 1/2*np.linalg.norm(res)**2 + sorted_L1_norm(b, Lambda)
    norm_res = dual_sorted_L1_norm(X.T@res, Lambda)
    res = res if norm_res<=1. or isnan(norm_res) or isinf(norm_res) else res/norm_res
    dual = 1/2*(np.linalg.norm(y)**2 - np.linalg.norm(y-res)**2)
    gap = primal - dual
    return primal, gap

@njit
def pattern(b, tol):
    sign = np.sign(b)
    perm = np.argsort(-np.abs(b))
    b = np.abs(b)[perm]
    epsilon = min(tol*(b[0]-b[-1])/len(b), tol)
    jump = b - np.append(b[1:],0) > epsilon
    # q = np.array([np.sum(jump[i:]) for i in range(len(b))]) # slow
    q = np.flip(np.cumsum(np.flip(jump))) # quicker
    m = np.arange(len(b))
    m[perm] = q * sign[perm]
    return m

@njit
def face_pattern(z, Lambda, rtol, atol):
    z = z / dual_sorted_L1_norm(z,Lambda)  # the dual sorted L1 norm of z is set to 1
    sign = np.sign(z)
    perm = np.argsort(-np.abs(z))
    z = np.abs(z)[perm]
    # r = np.cumsum(z) / np.cumsum(Lambda)
    # l = np.isclose(r, 1, rtol=tol)
    l = np.isclose(np.cumsum(z), np.cumsum(Lambda), rtol, atol) # new (consistent with gamma_split)
    # q = np.array([np.sum(l[i:]) for i in range(len(z))]) # slow
    q = np.flip(np.cumsum(np.flip(l)))  # quicker
    m = np.arange(len(z))
    m[perm] = q * sign[perm]
    return m

@njit
def pattern_matrix(m):
    k = np.max(np.abs(m))
    U_m = np.empty((len(m),k))
    for j in range(k):
        U_m[:,j] = np.sign(m)*(np.abs(m) == k-j) 
    return U_m

@njit
def affine_components(X, y, Lambda, m):
    X_tilde = X @ pattern_matrix(m) # clustered matrix
    m = - np.sort(-np.abs(m)) # m sorted by decreasing absolute value
    Lambda_tilde = pattern_matrix(m).T @ Lambda # clustered parameter
    P = X_tilde.T @ X_tilde
    # s(gamma) = a_s * gamma + b_s
    a_s = - np.linalg.solve(P, Lambda_tilde)
    b_s = np.linalg.solve(P, X_tilde.T @ y)
    # g(gamma) = a_g * gamma + b_g
    a_g = - X.T @ X_tilde @ a_s
    b_g = X.T @ (y - X_tilde @ b_s)
    return a_s, b_s, a_g, b_g

