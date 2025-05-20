import numpy as np

x=np.array([1,2,3,4,5])
y=np.array([2,4,6,8,10])
x_mean=np.mean(x)
y_mean=np.mean(y)
b1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b0=y_mean-b1 * x_mean

print(f"Slope :{b1}")
print(f"y-Intercept:{b0}")

predict=b0+b1*x
print("predicted values :",predict)