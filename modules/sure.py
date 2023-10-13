import numpy as np
from modules.utils import pattern, pattern_matrix, affine_components, affine_components_LASSO

def SURE_SLOPE(X,y,Sol,sigma2, tol):
    A=np.linalg.norm(y-X @ Sol, ord=2)**2
    B=2*sigma2*np.linalg.norm(pattern(Sol, tol),ord=np.inf)-X.shape[0]*sigma2
    SURE=A+B
    return(SURE)

def critical_points(X,y,m,a_s,b_s):
    Xtilde=X @ pattern_matrix(m)
    N=np.transpose(a_s) @ np.transpose(Xtilde) @ (y -Xtilde @ b_s)
    D=np.linalg.norm(Xtilde @ a_s, ord=2)**2
    c=N/D
    return(c)

def min_SURE(X,y,Lambda,Gamma,M,sigma2, tol):
    SURE=list()
    Critical = list()
    Solution = list()
    kmax=len(Gamma)-1  
    for k in range(kmax):
        m = M[k] 
        a_s,b_s, *_ = affine_components(X, y, Lambda, m)
        c=critical_points(X, y, m, a_s, b_s)
        gamma_0=Gamma[k]
        gamma_1=Gamma[k+1]
        Sol0=pattern_matrix(m) @ (a_s * gamma_0 + b_s )
        sure=SURE_SLOPE(X,y,Sol0,sigma2,tol)
        SURE.append(sure)
        Critical.append(gamma_0)
        Solution.append(Sol0)
        Sol1=pattern_matrix(m) @ (a_s * gamma_1 + b_s ) 
        if c<gamma_0 and c>gamma_1:
            a= (c-gamma_1)/(gamma_0-gamma_1)
            Solc=a*Sol0+(1-a)*Sol1
            sure=SURE_SLOPE(X,y,Solc,sigma2,tol)
            SURE.append(sure)
            Critical.append(c)
            Solution.append(Solc)
    m = M[-1]
    a_s,b_s, *_ = affine_components(X, y, Lambda, m)
    gamma_0=Gamma[-1]
    Sol0=pattern_matrix(m) @ (a_s * gamma_0 + b_s )
    sure=SURE_SLOPE(X,y,Sol0,sigma2, tol)
    SURE.append(sure)
    Critical.append(gamma_0)
    Solution.append(Sol0)
    return Critical, SURE, Solution

def SURE_LASSO(X,y,Sol,sigma2,tol):
    A=np.linalg.norm(y-X@Sol,ord=2)**2
    B=2*sigma2*np.sum(np.abs(Sol)>tol)-X.shape[0]*sigma2
    SURE=A+B
    return(SURE)

def critical_points_LASSO(X,y,s,a_s,b_s):
    I=s!=0
    N=np.transpose(a_s) @ np.transpose(X[:,I]) @ (y -X[:,I] @ b_s)
    D=np.linalg.norm(X[:,I] @ a_s, ord=2)**2
    c=N/D
    return(c)

def min_SURE_LASSO(X,y,Gamma,S,sigma2, tol):
    SURE=list()
    Critical = list()
    Solution = list()
    kmax=len(Gamma)-1  
    for k in range(kmax):
        s = S[k] 
        a_s,b_s, *_ = affine_components_LASSO(X, y, s)
        c = critical_points_LASSO(X, y, s, a_s, b_s)
        gamma_0=Gamma[k]
        gamma_1=Gamma[k+1]
        Sol0 = np.zeros(X.shape[1],dtype=float)
        Sol1 = np.zeros(X.shape[1],dtype=float)
        I=s!=0
        Sol0[I] = a_s * gamma_0 + b_s 
        sure = SURE_LASSO(X,y,Sol0,sigma2, tol)
        SURE.append(sure)
        Critical.append(gamma_0)
        Solution.append(Sol0)
        Sol1[I]=a_s * gamma_1 + b_s  
        if c<gamma_0 and c>gamma_1:
            a = (c-gamma_1)/(gamma_0-gamma_1)
            Solc = a*Sol0 + (1-a)*Sol1
            sure=SURE_LASSO(X,y,Solc,sigma2, tol)
            SURE.append(sure)
            Critical.append(c)
            Solution.append(Solc)
    s = S[-1]
    a_s,b_s, *_ = affine_components_LASSO(X, y, s)
    gamma_0=Gamma[-1]
    I=s!=0
    Sol0[I] = a_s * gamma_0 + b_s 
    sure = SURE_LASSO(X,y,Sol0,sigma2, tol)
    SURE.append(sure)
    Critical.append(gamma_0)
    Solution.append(Sol0)
    return Critical, SURE, Solution