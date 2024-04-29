import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = [ -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
y = [ 90, 77, 58, 35, 23, 15, 9, 5, 3, 0, 3, 5, 9, 15, 23, 35, 58, 77, 90]

#x = [ -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
#y = [1.2, 4.2, 6.7, 8.3, 10.6, 11.7, 13.5, 14.5, 15.7, 16.1, 16.6, 16.0, 15.4, 14.4, 14.2, 12.7, 10.3, 8.6, 6.1, 3.9, 2.1]

x = np.asarray(x)
y = np.asarray(y)
plt.plot(x, y, 'o')
#plt.show()

def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y

parameters, covariance = curve_fit(Gauss, x, y)

fit_A = parameters[0]
fit_B = parameters[1]
print(fit_A)
print(fit_B)

def cos(x, D, E):
    y = D*np.cos(E*x)
    return y

parameters, covariance = curve_fit(cos, x, y)
fit_D = parameters[0]
fit_E = parameters[1]

cos_fit = cos(x, fit_D, fit_E)

plt.plot(x, y, 'o', label='data')
plt.plot(x, cos_fit, '-', label='fit')
#plt.show()

guess = [1, 0.1105]
parameters, covariance = curve_fit(cos, x, y, p0=guess)
fit_D = parameters[0]
fit_E = parameters[1]

cos_fit = cos(x, fit_D, fit_E)

plt.plot(x, y, 'o', label='data')
plt.plot(x, cos_fit, '-', label='fit')
plt.show()
#plt.savefig('03-cosine_fit2.png')

