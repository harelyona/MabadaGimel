import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data
data = pd.read_csv('data/I_vs_V.csv')

# Fit ohms law: V = R * I
def ohms_law(I, R):
    return R * I

# Extract current and voltage
I_str = data['Current (I)'].values
V1 = data['V1 (Volts)'].values
V2 = data['V2 (Volts)'].values

# Convert current from string to float in Amperes
def convert_current(curr_str):
    if 'uA' in curr_str:
        return float(curr_str.replace(' uA', '')) * 1e-6
    elif 'mA' in curr_str:
        return float(curr_str.replace(' mA', '')) * 1e-3
    else:
        return float(curr_str)
    
I = np.array([convert_current(i) for i in I_str])

I_mask = I<0.004
V1 = V1[I_mask]
I = I[I_mask]

# Fit the data to find resistance R
popt, pcov = curve_fit(ohms_law, I, V1)
R_fit = popt[0]
print(f"Fitted Resistance R: {R_fit:.2f} Ohms")

# Generate fitted line data
I_fit = np.linspace(min(I), max(I), 1000)
V_fit = ohms_law(I_fit, R_fit)

# Plot the data and the fitted line
plt.figure(figsize=(10, 6))
plt.plot(I, V1, 'o-', label='I vs V1', markersize=6)
plt.plot(I_fit, V_fit, 'r--', label=f'Fitted Line: V = {R_fit:.2f} * I')
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs Current')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig('I_vs_V.png', dpi=300)
plt.show()