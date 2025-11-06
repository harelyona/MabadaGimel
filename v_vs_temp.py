import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/v_vs_temp_even_better/Non-Shifted Results Clean.csv')

# Create figure with one subplot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot voltage vs time on the primary y-axis
color1 = 'tab:blue'
ax1.plot(data['Time'], data['Voltage '], color=color1, linewidth=1, label='Voltage')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Voltage (V)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# Create secondary y-axis for temperature
ax2 = ax1.twinx()
color2 = 'tab:red'
# Convert temperature from Celsius to Kelvin with precise conversion
temperature_kelvin = data['Temprature'] + 273.15
ax2.plot(data['Time'], temperature_kelvin, color=color2, linewidth=1, label='Temperature')
ax2.set_ylabel('Temperature (K)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Add title and legends
fig.suptitle('Voltage and Temperature vs Time', fontsize=14)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('v_vs_temp_plot.png', dpi=150, bbox_inches='tight')
plt.show()
