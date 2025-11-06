import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data
data = pd.read_csv('data/v_vs_temp_even_better/Non-Shifted Results Clean.csv')

# Filter data for time > 2000 seconds
data_filtered = data[data['Time'] > 2000].copy()


# Extract temperature and voltage
T = data_filtered['Temprature'].values  # Temperature in Celsius
V = data_filtered['Voltage '].values  # Voltage (related to mobility or carrier concentration)

# Convert temperature to Kelvin
T_kelvin = T + 273.15
# T_mask = T_kelvin < 225  
# T_kelvin = T_kelvin[T_mask]
# V = V[T_mask]

T_log = np.log(T_kelvin)
V_log = np.log(V)

def model_func(T,a, b):
    return a * T + b

params, covariance = curve_fit(model_func, T_log, V_log, p0=[1.5,0])
a,b = params
print(f"Fitted parameters: a = {a}, b = {b}")

x_range = np.linspace(min(T_log), max(T_log), 1000)
y_fit = model_func(x_range,a, b)

plt.figure(figsize=(10, 6))
plt.plot(T_log,V_log, 'o', label='Data', markersize=4)
plt.plot(x_range, y_fit, 'r-', label='Fitted Line')
# plt.plot(T_kelvin, V, 'o', label='Data', markersize=4)
plt.xlabel('Temperature (K)')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs Temperature')
plt.grid(True)
plt.legend()
plt.savefig('v_vs_temp_fit.png', dpi=300)
plt.show()
