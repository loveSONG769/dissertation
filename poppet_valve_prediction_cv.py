import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from random import randint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold,cross_val_score,KFold,train_test_split,cross_val_predict
from scipy import stats


#FC = pd.DataFrame({'x': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
#                   'y': [0, 3, 5, 9, 15, 23, 35, 58, 77, 90]})

x= [0,5.75,11.5,17.25,23,28.75,34.5,40.25,46,51.75,57.5,63.25,69,74.75,80.5,86.25,92,97.75,103.5,109.25,115,120.75,126.5,132.25,138,143.75,149.5,155.25,161,166.75,172.5,178.25,184,189.75,195.5,201.25,207]
y= [0,0.008164,0.016328,0.024492,0.04082,0.085722,0.18369,0.338806,0.518414,0.689858,0.849056,0.97968,1.093976,1.179698,1.249092,1.289912,1.310322,1.302158,1.273584,1.220518,1.14296,0.971516,0.840892,0.689858,0.53066,0.36738,0.216346,0.159198,0.089804,0.057148,0.044902,0.036738,0.028574,0.02041,0.012246,0.004082,0]


r = random.sample(range(0,36),7)
print(len(r))
print(r)
#print(F)

fit_x = []
for index, element in enumerate(x):
    if index not in r:
        fit_x.append(element)
#print(len(test_x))
print(len(fit_x))
test_x = []
test_x = (set(x)| set(fit_x)) - (set(x) & set(fit_x))
test_x = list(test_x)

print("test_x = ", test_x)

fit_y = []
test_y=[]
for index, element in enumerate(y):
    if index not in r:
        fit_y.append(element)
    else: test_y.append(element)
#test_y = []
#test_y = (set(y)| set(fit_y)) - (set(y) & set(fit_y))
test_y = list(test_y)
print("test_y = ", test_y)


#print(test_x)
#print(len(fit_x))

FC = pd.DataFrame({'x': fit_x,
                   'y': fit_y})

degree_1 = np.poly1d(np.polyfit(FC.x, FC.y, 1))
degree_2 = np.poly1d(np.polyfit(FC.x, FC.y, 2))
degree_3 = np.poly1d(np.polyfit(FC.x, FC.y, 3))
degree_4 = np.poly1d(np.polyfit(FC.x, FC.y, 4))
degree_5 = np.poly1d(np.polyfit(FC.x, FC.y, 5))
degree_6 = np.poly1d(np.polyfit(FC.x, FC.y, 6))
degree_7 = np.poly1d(np.polyfit(FC.x, FC.y, 7))
degree_8 = np.poly1d(np.polyfit(FC.x, FC.y, 8))
degree_9 = np.poly1d(np.polyfit(FC.x, FC.y, 9))
degree_10 = np.poly1d(np.polyfit(FC.x, FC.y, 10))
degree_11 = np.poly1d(np.polyfit(FC.x, FC.y, 11))
degree_12 = np.poly1d(np.polyfit(FC.x, FC.y, 12))
degree_13 = np.poly1d(np.polyfit(FC.x, FC.y, 13))
degree_14 = np.poly1d(np.polyfit(FC.x, FC.y, 14))



poly = np.linspace(0, 207, 100)
plt.scatter(FC.x, FC.y)

plt.plot(poly, degree_1(poly), color='pink')
plt.plot(poly, degree_2(poly), color='red')
plt.plot(poly, degree_3(poly), color='black')
plt.plot(poly, degree_4(poly), color='blue')
plt.plot(poly, degree_5(poly), color='orange')
plt.plot(poly, degree_6(poly), color='yellow')
plt.plot(poly, degree_7(poly), color='brown')
plt.plot(poly, degree_8(poly), color='olive')
plt.plot(poly, degree_9(poly), color='cyan')
plt.plot(poly, degree_10(poly), color='lime')
plt.plot(poly, degree_11(poly), color='gray')
plt.plot(poly, degree_12(poly), color='tan')
plt.plot(poly, degree_13(poly), color='violet')
plt.plot(poly, degree_14(poly), color='purple')

plt.title("Poppet Valve Opening Area Approximation 1-14 degree")
plt.xlabel("Crankshaft Angle Degree (degree)")
plt.ylabel("Valve Openinig Area (sq.in.)")
#plt.show()

def adj_R(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = 1- (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1))
    print(results)

adj_R(FC.x, FC.y, 1)
adj_R(FC.x, FC.y, 2)
adj_R(FC.x, FC.y, 3)
adj_R(FC.x, FC.y, 4)
adj_R(FC.x, FC.y, 5)
adj_R(FC.x, FC.y, 6)
adj_R(FC.x, FC.y, 7)
adj_R(FC.x, FC.y, 8)
adj_R(FC.x, FC.y, 9)
adj_R(FC.x, FC.y, 10)
adj_R(FC.x, FC.y, 11)
adj_R(FC.x, FC.y, 12)
adj_R(FC.x, FC.y, 13)
adj_R(FC.x, FC.y, 14)


poly = np.linspace(0, 207, 100)
plt.scatter(FC.x, FC.y)

plt.plot(poly, degree_14(poly))
plt.title("Poppet Valve Opening Area Approximation 14 degree")
plt.xlabel("Crankshaft Angle Degree (degree)")
plt.ylabel("Valve Openinig Area (sq.in.)")
#plt.show()



y_cv = degree_14
y_cvt = degree_14(test_x)
print("test_y = ", test_y)
print("y_cvt = ", y_cvt)
mse_cv = mean_squared_error(test_y, y_cvt)
print("mse_cv = ", mse_cv)




y_p = degree_14(x)
#print("y_p=",y_p)
degree=14
mse = mean_squared_error(y, y_p)
print("mean squared error= ",mse)

confidence = 0.95
n=len(y)
dof=n-degree-1
t=stats.t.ppf(confidence,dof)
print("t= ",t)

std_err=np.sqrt(np.sum((y-y_p)**2)/dof)
print("standard error= ",std_err)
margin_err=t*std_err
print("margin error = ", margin_err)
