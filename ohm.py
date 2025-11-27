import inspect
from typing import List
from gaussian import *
import pandas as pd

OHM_DATA_FILE = "data/ohm_law_data.csv"
def linear(x, a):
    return a * x

def square(x, a, b):
    return a * x**2 + b * x

def cubic(x, a, b):
    return a * x**3 + b * x



df = pd.read_csv(OHM_DATA_FILE)
df.sort_values(['Temperature_C', 'Voltage_V'], inplace=True)
temperatures = np.sort(df['Temperature_C'].unique())
voltage_max = df['Voltage_V'].max()
v_range = np.linspace(0, voltage_max)

def data_subset_by_temperature(data_frame, temperature):
    subset = data_frame[data_frame['Temperature_C'] == temperature]
    voltage = subset['Voltage_V'].values
    current = subset['Current_A'].values
    return voltage, current

def compute_r_squared(x, y, model_func, params):
    residuals = y - model_func(x, *params)
    ss_res = np.sum(residuals ** 2)  # Residual Sum of Squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total Sum of Squares
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def plot_current_vs_voltage(fit_func, specific_temperature: List[float]=None):
    r_square = {}
    temperatures_to_plot = temperatures if specific_temperature is None else specific_temperature
    for T in temperatures_to_plot:
        voltage, current = data_subset_by_temperature(df, T)
        popt, pcov = curve_fit(fit_func, voltage, current)
        sc = plt.scatter(voltage, current, label=f"{T:<10}", s=DATA_SIZE)
        point_color = sc.get_facecolors()[0]
        x_range = np.linspace(0, voltage.max())
        plt.plot(x_range, fit_func(x_range, *popt), color=point_color)
        r_square[T] = compute_r_squared(voltage, current, fit_func, popt)
    plot_config("Voltage (V)", "Current (A)", "Current vs Voltage^2 at Different Temperatures")
    plt.show()
    print("R^2 values for each temperature:")
    for T, r2 in r_square.items():
        print(f"Temperature {T} C: R^2 = {r2:.4f}")


def plot_parameters_vs_temperature(func):
    # 1. Extract parameter names from the function signature
    sig = inspect.signature(func)
    # We skip the first parameter (index 0) because it is the independent variable (x/voltage)
    param_names = list(sig.parameters.keys())[1:]

    # Initialize storage
    num_params = len(param_names)
    parameters = np.zeros((num_params, len(temperatures)))

    # 2. Fill Data
    # Using enumerate is cleaner than np.where
    for idx, T in enumerate(temperatures):
        voltage, current = data_subset_by_temperature(df, T)
        popt, _ = curve_fit(func, voltage, current)
        parameters[:, idx] = popt  # Fill the whole column at once

    # 3. Plot with Labels
    for i in range(num_params):
        # Use param_names[i] for the label
        plt.plot(temperatures, parameters[i, :], marker='o', label=param_names[i])
        plt.legend(loc='best')
        plt.show()
    plot_config("Temperature (C)", "Fitted Parameters", "Fitted Parameters vs Temperature")
    plt.show()

def plot_conductivity_vs_temperature():
    conductivities = []
    for T in temperatures:
        voltage, current = data_subset_by_temperature(df, T)
        popt, _ = curve_fit(cubic, voltage, current)
        R = popt[1]  # Since I = (1/R) * V
        conductivities.append(R)
    plt.plot(temperatures, conductivities, marker='o')
    plot_config("Temperature (C)", "Conductivity (S/m)", "Conductivity vs Temperature")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

#V = IR
#I = V/R
if __name__ == "__main__":

    plot_current_vs_voltage(cubic)
    plot_parameters_vs_temperature(cubic)