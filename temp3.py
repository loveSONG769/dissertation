import numpy as np
from scipy import integrate


def h_c(x):
    return (1 - x ** 2) ** 0.5
    
N = 1000
x = np.linspace(-1, 1, N)
dx = 2. / N;
y = h_c(x)
area =sum( dx * y)
h_p, err = integrate.quad(h_c, -1,1) 
print (h_p * 2)
