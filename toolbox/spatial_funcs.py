import numpy as np
from scipy.stats import ncx2

def cascade(z, Sigma_t, P):
    inside  = (1 - np.exp(-Sigma_t * z)) / Sigma_t
    outside = (np.exp(Sigma_t * (P - z)) - np.exp(-Sigma_t * z)) / Sigma_t
    return np.where(z <= P, inside, outside)

def cascade_lateral(r, R, sigma):
    a = np.asarray(r) / sigma
    b = R / sigma
    # f(r) = 1 - Q1(a,b) = CDF(ncx2; df=2, nc=a^2) at b^2
    return ncx2.cdf(b**2, df=2, nc=a**2)
