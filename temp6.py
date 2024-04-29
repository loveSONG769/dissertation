import numpy as np
import matplotlib.pyplot as plt
 
 
x_data = [0,5.75,11.5,17.25,23,28.75,34.5,40.25,46,51.75,57.5,63.25,69,74.75,80.5,86.25,92,97.75,103.5,109.25,115,120.75,126.5,132.25,138,143.75,149.5,155.25,161,166.75,172.5,178.25,184,189.75,195.5,201.25,207]

y_data = [0,0.002,0.004,0.006,0.01,0.021,0.045,0.083,0.127,0.169,0.208,0.24,0.268,0.289,0.306,0.316,0.321,0.319,0.312,0.299,0.28,0.238,0.206,0.169,0.13,0.09,0.053,0.039,0.022,0.014,0.011,0.009,0.007,0.005,0.003,0.001,0]
 
poly = np.polyfit(x_data, y_data, deg = 15)

print(poly)
 
plt.plot(x_data, y_data, 'o')
plt.plot(x_data, np.polyval(poly, x_data))
plt.show()