import inspect
import os.path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from gaussian import *
import pandas as pd

OHM_DATA_FILE = "data/ohm_law_data.csv"
def linear(x, a):
    return a * x

def square(x, a, b):
    return a * x**2 + b * x

def cubic(x, a, b):
    return a * x**3 + b * x

def conductivity_hot_function(x, A):
    return A * (x)**(1.5)



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


def plot_current_vs_voltage(fit_func, specific_temperature: List[float]=None, output_name : str=None):
    temperatures_to_plot = temperatures if specific_temperature is None else specific_temperature
    for T in temperatures_to_plot:
        voltage, current = data_subset_by_temperature(df, T)
        popt, pcov = curve_fit(fit_func, voltage, current)
        r_square = compute_r_squared(voltage, current, fit_func, popt)
        # 1. Capture the return value (the container)
        container = plt.errorbar(voltage, current, xerr=0.05, yerr=0.0005,
                                 label=rf"T=${T:<10}^{{\circ}}, r^2 = {r_square:.2f}$",
                                 fmt='o', capsize=2, ms=4)

        # 2. Extract the color from the first element (the main line/marker)
        data_color = container[0].get_color()
        x_range = np.linspace(0, voltage.max())


        plt.plot(x_range, fit_func(x_range, *popt), color=data_color)

    plot_config("Voltage (V)", "Current (A)")
    if output_name:
        plt.savefig(PLOTS_DIRECTORY + output_name + ".png")
    plt.show()


def plot_parameters_vs_temperature(func, save_name=None):
    # 1. Extract parameter names from the function signature
    sig = inspect.signature(func)
    # We skip the first parameter (index 0) because it is the independent variable (x/voltage)
    param_names = list(sig.parameters.keys())[1:]

    # Initialize storage
    num_params = len(param_names)
    parameters = np.zeros((num_params, len(temperatures)))
    errors = np.zeros((num_params, len(temperatures)))
    # 2. Fill Data
    # Using enumerate is cleaner than np.where
    for idx, T in enumerate(temperatures):
        voltage, current = data_subset_by_temperature(df, T)
        popt, pcov = curve_fit(func, voltage, current)
        parameters[:, idx] = popt  # Fill the whole column at once
        errors[:, idx] = np.sqrt(np.diag(pcov))

    # 3. Plot with Labels
    for i in range(num_params):
        # Use param_names[i] for the label

        plt.errorbar(temperatures, parameters[i, :], errors[i, :], fmt='-o', label=f"{param_names[i]} Value", capsize=5)
        plt.legend(loc='best')
        plot_config("Temperature (C)", "Fitted Parameter")
        if save_name:
            plt.savefig(PLOTS_DIRECTORY + save_name)
        plt.show()
        break # Only plot the first parameter for brevity

def plot_conductivity_vs_temperature():
    conductivities = []
    for T in temperatures:
        voltage, current = data_subset_by_temperature(df, T)
        popt, _ = curve_fit(cubic, voltage, current)
        conductivity = popt[1]  # Since I = (1/R) * V
        conductivities.append(conductivity)
    plt.scatter(temperatures, conductivities, marker='o', label="measured conductivity")
    temperature_range = np.linspace(min(temperatures), max(temperatures))
    popt, pcov = curve_fit(conductivity_hot_function, temperatures, conductivities)
    plt.plot(temperature_range, conductivity_hot_function(temperature_range, *popt), label="fit")
    plot_config("Temperature (C)", "Conductivity (S/m)", )
    plt.show()

#V = IR
#I = V/R
display_temperature = [23, 85, 95, 100, 117,135]

def main_plot_all_ohms():
    plot_current_vs_voltage(cubic, specific_temperature=display_temperature, output_name="ohm_law_cubic_partial")
    plot_current_vs_voltage(linear, specific_temperature=display_temperature, output_name="ohm_law_linear_partial")
    plot_current_vs_voltage(cubic, output_name="ohm_law_cubic_full")
    plot_current_vs_voltage(linear, output_name="ohm_law_linear_full")


if __name__ == "__main__":
    plot_conductivity_vs_temperature()
