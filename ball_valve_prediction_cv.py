import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from random import randint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold,cross_val_score,KFold,train_test_split,cross_val_predict
from scipy import stats




x = np.linspace(0,90,200)
#x=x+np.random.normal(0,.3,x.shape)



y = [7.942e-02, 1.934e-01, 3.131e-01, 4.241e-01,
 5.434e-01, 6.532e-01, 7.687e-01, 8.830e-01,
 9.922e-01, 1.168e+00, 1.221e+00, 1.365e+00,
 1.468e+00, 1.542e+00, 1.677e+00, 1.742e+00,
 1.937e+00, 2.073e+00, 2.140e+00, 2.217e+00,
 2.315e+00, 2.414e+00, 2.614e+00, 2.774e+00,
 2.875e+00, 2.987e+00, 3.100e+00, 3.254e+00,
 3.349e+00, 3.465e+00, 3.602e+00, 3.739e+00,
 3.848e+00, 3.958e+00, 4.118e+00, 4.210e+00,
 4.383e+00, 4.547e+00, 4.652e+00, 4.778e+00,
 4.935e+00, 5.014e+00, 5.204e+00, 5.344e+00,
 5.427e+00, 5.620e+00, 5.775e+00, 5.941e+00,
 6.049e+00, 6.298e+00, 6.369e+00, 6.591e+00,
 6.655e+00, 6.861e+00, 6.989e+00, 7.178e+00,
 7.320e+00, 7.433e+00, 7.638e+00, 7.866e+00,
 7.976e+00, 8.168e+00, 8.323e+00, 8.551e+00,
 8.641e+00, 8.824e+00, 9.050e+00, 9.249e+00,
 9.431e+00, 9.687e+00, 9.826e+00, 1.002e+01,
 1.073e+01, 1.044e+01, 1.065e+01, 1.087e+01,
 1.140e+01, 1.172e+01, 1.156e+01, 1.139e+01,
 1.223e+01, 1.228e+01, 1.253e+01, 1.279e+01,
 1.376e+01, 1.383e+01, 1.360e+01, 1.328e+01,
 1.487e+01, 1.417e+01, 1.477e+01, 1.568e+01,
 1.530e+01, 1.562e+01, 1.606e+01, 1.670e+01,
 1.624e+01, 1.740e+01, 1.747e+01, 1.784e+01,
 1.852e+01, 1.832e+01, 1.902e+01, 1.993e+01,
 1.945e+01, 2.098e+01, 2.072e+01, 2.147e+01,
 2.114e+01, 2.261e+01, 2.259e+01, 2.319e+01,
 2.379e+01, 2.471e+01, 2.464e+01, 2.568e+01,
 2.543e+01, 2.649e+01, 2.687e+01, 2.746e+01,
 2.836e+01, 2.867e+01, 2.929e+01, 2.993e+01,
 3.058e+01, 3.144e+01, 3.192e+01, 3.281e+01,
 3.321e+01, 3.462e+01, 3.474e+01, 3.548e+01,
 3.613e+01, 3.730e+01, 3.777e+01, 3.836e+01,
 3.976e+01, 4.078e+01, 4.100e+01, 4.164e+01,
 4.249e+01, 4.345e+01, 4.442e+01, 4.570e+01,
 4.639e+01, 4.729e+01, 4.801e+01, 4.833e+01,
 4.976e+01, 5.090e+01, 5.175e+01, 5.291e+01,
 5.328e+01, 5.445e+01, 5.563e+01, 5.661e+01,
 5.740e+01, 5.830e+01, 5.960e+01, 6.060e+01,
 6.171e+01, 6.282e+01, 6.362e+01, 6.433e+01,
 6.544e+01, 6.655e+01, 6.765e+01, 6.825e+01,
 6.915e+01, 7.024e+01, 7.162e+01, 7.270e+01,
 7.337e+01, 7.402e+01, 7.547e+01, 7.640e+01,
 7.762e+01, 7.862e+01, 7.910e+01, 7.996e+01,
 8.041e+01, 8.133e+01, 8.243e+01, 8.350e+01,
 8.324e+01, 8.476e+01, 8.534e+01, 8.529e+01,
 8.611e+01, 8.748e+01, 8.772e+01, 8.821e+01,
 8.876e+01, 8.997e+01, 8.942e+01, 8.932e+01,
 8.947e+01, 9.066e+01, 9.029e+01, 9.016e+01]

r = random.sample(range(0,200),20)#[random.randint(0,200) for i in range(20)]
#print(len(r))
#print("r=",r)

fit_x = []
for index, element in enumerate(x):
    if index not in r:
        fit_x.append(element)
#print(len(test_x))
#print(len(fit_x))
test_x = []
test_x = (set(x)| set(fit_x)) - (set(x) & set(fit_x))
test_x = list(test_x)
#print(test_x)



fit_y = []
for index, element in enumerate(y):
    if index not in r:
        fit_y.append(element)
test_y = []
test_y = (set(y)| set(fit_y)) - (set(y) & set(fit_y))
#print("test_y = ", test_y)
test_y = list(test_y)

#print("x = ", x)
#print("test_x = ",test_x)

print(len(test_x))
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



poly = np.linspace(0, 90, 50)
plt.scatter(FC.x, FC.y)

plt.plot(poly, degree_1(poly), color='pink')
plt.plot(poly, degree_2(poly), color='red')
plt.plot(poly, degree_3(poly), color='black')
plt.plot(poly, degree_4(poly), color='blue')
plt.plot(poly, degree_5(poly), color='orange')
plt.plot(poly, degree_6(poly), color='yellow')
plt.title("Ball Valve Flow Coefficient Approximation 1-6 degree")
plt.xlabel("Valve Opening Degree (degree)")
plt.ylabel("Flow Coefficient (%)")
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


poly = np.linspace(0, 90, 50)
plt.scatter(FC.x, FC.y)

plt.plot(poly, degree_5(poly))
plt.title("Ball Valve Flow Coefficient Approximation")
plt.xlabel("Valve Opening Degree (degree)")
plt.ylabel("Flow Coefficient (%)")
#plt.show()


#x=x.reshape((200,1))

#x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state=0)
#training_error=[]
#cross_validation_error=[]
#for d in range(1,7):
#    x_poly_train=PolynomialFeatures(degree=d).fit_transform(x_train)
#    x_poly_test=PolynomialFeatures(degree=d).fit_transform(x_test)
#    lr=LinearRegression(fit_intercept=False)
#    model=model.fit(x_poly_train,y_train)
#    y_train_pred=model.predict(x_poly_train)
#    mse_train=mean_squared_error(y_train,y_train_pred)
#    cve=cross_validate(lr,x_poly_train,y_train,scoring='neg_mean_squared_error',cv=5,return_train_score=True)
#    training_error.append(mse_train)
#    cross_validation_error.append(np.mean(np.absolute(cve['test_score'])))
#fig,ax=plt.subplots(figsize=(6,6))
#ax.plot(range(1,maxdegree),cross_validation_error)
#ax.set_xlabel('Degree',fontsize=20)
#ax.set_ylabel('MSE',fontsize=20)
#ax.set_title('MSE VS Degree',fontsize=25)






y_cv = degree_5
y_cvt = degree_5(test_x)
#print("test_y = ", test_y)
#print("y_cvt = ", y_cvt)
mse_cv = mean_squared_error(test_y, y_cvt)
print("mse_cv = ", mse_cv)




y_p = degree_5(x)
#print("y_p=",y_p)
degree=5
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

poly = np.linspace(0, 90, 50)
#plt.scatter(FC.x, FC.y)

plt.plot(poly, degree_5(poly), color='black')
plt.plot(poly, degree_5(poly)+margin_err, color='pink')
plt.plot(poly, degree_5(poly)-margin_err, color='pink')
plt.title("Ball Valve Flow Coefficient Approximation")
plt.xlabel("Valve Opening Degree (degree)")
plt.ylabel("Flow Coefficient (%)")
#plt.show()
