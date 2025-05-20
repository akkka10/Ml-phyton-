import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


np.random.seed(17)
data = np.cumsum(np.random.randn(100))
dataSeries = pd.Series(data)

plt.figure(figsize=(10,5))
plt.plot(dataSeries, label="Original Data")
plt.title("Original Data")
plt.legend()
plt.show()

model = ARIMA(dataSeries, order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(steps=10)

plt.figure(figsize=(10,5))
plt.plot(dataSeries, label="Original Data")
plt.plot(range(len(dataSeries), len(dataSeries) + 10), forecast, label="Forecast", color="red")
plt.title("Original Data and Forecast")
plt.show()
plt.legend()