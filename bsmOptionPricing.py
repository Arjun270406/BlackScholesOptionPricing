import numpy as np
from scipy.stats import norm


S0 = 100
K = 100
r = 0.05
T = 1
Cmarket = 10


def black_scholes_call(S0, K, r, vol, T):
    d1 = (np.log(S0/K) + (r + 0.5*(vol**2))*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    return S0*norm.cdf(d1) - K*np.exp(-1*r*T)*norm.cdf(d2)


def black_scholes_put(S0, K, r, vol, T):
    d1 = (np.log(S0/K) + (r + 0.5*(vol**2))*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    return K*np.exp(-1*r*T)*norm.cdf(-1*d2) - S0*norm.cdf(-1*d1)


def f(vol):
    return black_scholes_call(S0, K, r, vol, T) - Cmarket


def df(vol):
    d1 = (np.log(S0/K) + (r + 0.5*(vol**2))*T)/(vol*np.sqrt(T))
    return S0*norm.pdf(d1)*np.sqrt(T)


def impliedVolatility(a, iter):
    while iter>0:               #Use Newton-Raphson on BS eqn to compute implied volatility
        a = a - f(a)/df(a)
        iter-=1

    return a


iv = impliedVolatility(1, 3)
print(iv)
