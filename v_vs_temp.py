import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data
# data = pd.read_csv('data/v_vs_temp/v_vs_temp_3_11.csv')
data = pd.read_csv('data/v_vs_temp_even_better/Non-Shifted Results Clean.csv')

# Extract temperature and voltage
# T = data['Temperature'].values  
# V = data['Voltage 1'].values  
T = data['Temprature'].values
V = data['Voltage '].values
time = data['Time'].values

# Convert temperature to Kelvin
T_kelvin = T + 273.15

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(time, T_kelvin, 'o', label='T', markersize=4, color='orange')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Temperature (K)', color='orange')
ax1.tick_params(axis='y', labelcolor='orange')

ax2 = ax1.twinx()
ax2.plot(time, V, 'o', label='V', markersize=4, color='blue')
ax2.set_ylabel('Voltage (V)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Voltage and Temp vs time')
ax1.grid(True)
fig.legend(loc='upper right')
plt.savefig('v_vs_temp.png', dpi=300)
plt.show()
