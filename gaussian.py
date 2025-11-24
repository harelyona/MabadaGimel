import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Tuple, Optional
import matplotlib.pyplot as plt

PLOTS_DIRECTORY = "plots/"
CSV_VOLT_COLUMN = 2
CSV_ACTIVATION_VOLTAGE_COLUMN = 1
CSV_TIME_COLUMN = 0
FIT_COLOR = "blue"
DADA_COLOR = "black"
ERROR_BAR_COLOR = "red"
DATA_SIZE = 7
CAPSIZE = 0.5
d = 0.0027
ACTIVATION_VOLTAGE_THRESHOLD = 2
INTENSITY_THRESHOLD = 1.1
haynes_shockley_dir = "data/Haynes-Shockley"
file1 = f"{haynes_shockley_dir}/Vs_28.3_Vl_27.7_d_2.7.csv"
file1_time_mask = (0.0000125, 0.000032)
file2 = f"{haynes_shockley_dir}/Vs_35.8_Vl_27.7_d_2.7.csv"
file2_time_mask = (0.0000, 0.000025)
file3 = f"{haynes_shockley_dir}/Vs_44_Vl_27.7_d_2.7.csv"
file3_time_mask = (0.00004, 0.0000575)
file4 = f"{haynes_shockley_dir}/Vs_50_Vl_27.7_d_2.7.csv"
file4_time_mask = (0.000043, 0.00006)
DATA_FILES = [file1, file2, file3, file4]
MASKS = {file1: file1_time_mask, file2: file2_time_mask, file3: file3_time_mask, file4: file4_time_mask}


def gaussian(x, A, mu, sigma, D):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2)) + D


def extract_data(file_path: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Extracts time and intensity data from a CSV file.
    Optionally filters the data to a specified range,
    drops any NaN values, and returns clean NumPy arrays.
    """
    df = pd.read_csv(file_path)
    intensities_series = pd.to_numeric(df.iloc[1:, CSV_VOLT_COLUMN], errors='coerce')
    time_series = pd.to_numeric(df.iloc[1:, CSV_TIME_COLUMN], errors='coerce')
    activation_lazer_series = pd.to_numeric(df.iloc[1:, CSV_ACTIVATION_VOLTAGE_COLUMN], errors='coerce')
    first_time_idx_lazer_on = (activation_lazer_series < ACTIVATION_VOLTAGE_THRESHOLD).idxmax()
    t0 = time_series[first_time_idx_lazer_on]


    data_df = pd.DataFrame({'time': time_series, 'intensities': intensities_series}).dropna()
    time_mask = pd.Series(True, index=data_df.index)

    if min_val is not None:
        time_mask = time_mask & (data_df['time'] > min_val)

    if max_val is not None:
        time_mask = time_mask & (data_df['time'] < max_val)
    mask = (intensities_series < INTENSITY_THRESHOLD) & (time_series >= t0) & time_mask
    time_filtered = data_df.loc[mask, 'time']
    intensities_filtered = data_df.loc[mask, 'intensities']
    time_values = time_filtered.values
    time_values = time_values
    intensities_values = intensities_filtered.values
    #intensities_values = fix_linear_drift(time_values, intensities_values)

    return time_values, intensities_values



def plot_v_vs_time(time: np.ndarray, intensities: np.ndarray, uncertainty: float, save:bool = False) -> Tuple[np.ndarray, np.ndarray]:
    params, cov_mat = curve_fit(gaussian, time, intensities, maxfev=99999, p0=[max(intensities), time[np.argmax(intensities)], 1e-5, min(intensities)])
    x_range = np.linspace(min(time), max(time), 1000)
    fitted_curve = gaussian(x_range, *params)
    A, mu, sigma, D = params
    sigma=abs(sigma)
    A_error, mu_error, sigma_error, D_error = np.sqrt(np.diag(cov_mat))
    print(f"A={A:.2e} ± {A_error:.2e}")
    print(f"mu={mu:.2e} ± {mu_error:.2e}")
    print(f"sigma={sigma:.2e} ± {sigma_error:.2e}")
    print(f"D={D:.2e} ± {D_error:.2e}")
    plt.plot(x_range, fitted_curve, label="gaussian fit", color=FIT_COLOR)
    plt.errorbar(time, intensities, yerr=uncertainty, label='Data', color=DADA_COLOR, fmt='o', capsize=CAPSIZE, markersize=1, elinewidth=2, ecolor=ERROR_BAR_COLOR)
    plot_config("Time", "Intensity")
    if save:
        plt.savefig(f"{PLOTS_DIRECTORY}intensity_vs_time.png")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()
    return params, cov_mat

def plot_config(x_label: str, y_label: str, plot_title:str = None) -> None:
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Only show legend if there are labeled artists (labels that don't start with '_')
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()

def linear_fit(x: np.ndarray, y: np.ndarray, show: bool=False) -> Tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    if show:
        print(f"Slope={slope:.2e}, intercept={intercept:.2e}")
        x_range = np.linspace(min(x), max(x), 1000)
        plt.plot(x_range, slope * x_range + intercept, label='Linear Fit')
        #plt.scatter(x, y, label='Data', s=DATA_SIZE, color=DADA_COLOR)
        plot_config(plot_title="Slope", x_label="x", y_label="y")
        plt.show()
    return slope, intercept


def plot_all_data(folder: str) -> None:
    for file_path in os.listdir(folder):
        full_file_path = os.path.join(folder, file_path)
        time, intensities = extract_data(full_file_path, max_val=0.0001)
        match = re.search(r'Vs_([\d\.]+)_Vl_([\d\.]+)_d_([\d\.]+)\.csv$', file_path)
        vs_value = float(match.group(1))
        vl_value = float(match.group(2))
        d_value = float(match.group(3)) / 1000  # Convert mm to m
        slope, intercept = linear_fit(*extract_data(full_file_path, max_val=0), show=False)
        plt.scatter(time, intensities, label=f"Vs={vs_value}, Vl={vl_value}, d={d_value}", s=DATA_SIZE)
    plot_config("Time", "Intensity")
    plt.show()

def fix_linear_drift(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    slope, intercept = linear_fit(x_data, y_data)
    drift = slope * x_data + intercept
    corrected_y = y_data - drift
    return corrected_y

def plot_data(file:str):
    time, intensities = extract_data(file)
    plt.scatter(time, intensities, s=DATA_SIZE)
    plt.show()

def parse_file_parameters(file_name: str) -> Tuple[float, float, float]:
    match = re.search(r'Vs_([\d\.]+)_Vl_([\d\.]+)_d_([\d\.]+)\.csv$', file_name)
    vs_value = float(match.group(1))
    vl_value = float(match.group(2))
    d_value = float(match.group(3)) / 1000  # Convert mm to m
    return vs_value, vl_value, d_value

def plot_mu_vs_Vs() -> None:
    Vs_values = []
    mu_values = []
    for file in DATA_FILES:
        time, intensities = extract_data(file, *MASKS[file])
        params, _ = curve_fit(gaussian, time, intensities, maxfev=99999, p0=[max(intensities), time[np.argmax(intensities)], 1e-5, min(intensities)])
        _, mu, _, _ = params
        vs_value, _, _ = parse_file_parameters(file)
        Vs_values.append(vs_value)
        mu_values.append(mu)
    plt.scatter(Vs_values, mu_values, s=DATA_SIZE, color=DADA_COLOR, label="Data")
    plot_config("Source Voltage (V)", "Mobility (m^2/Vs)")
    plt.show()

def plot_sigma_vs_Vs() -> None:
    Vs_values = []
    sigma_values = []
    uncertainty_values = []
    for file in DATA_FILES:
        time, intensities = extract_data(file, *MASKS[file])
        uncertainty = np.std(intensities)

        params, cov_mat = curve_fit(gaussian, time, intensities, maxfev=99999, p0=[max(intensities), time[np.argmax(intensities)], 1e-5, min(intensities)])
        _, _, sigma, _ = params
        sigma=abs(sigma)
        sigma_error = cov_mat[2][2]**0.5
        sigma_values.append(sigma)
        uncertainty_values.append(sigma_error)
        vs_value, _, _ = parse_file_parameters(file)
        Vs_values.append(vs_value)
    plt.scatter(Vs_values, sigma_values, s=DATA_SIZE)
    plot_config("Source Voltage (V)", "Sigma (s)")
    plt.show()

def plot_gaussians():
    for file in DATA_FILES:
        time, intensities = extract_data(file, *MASKS[file])
        vs, _, _ = parse_file_parameters(file)
        plt.scatter(time, intensities, s=DATA_SIZE,label=f"Vs={vs}")
    plot_config("Time (s)", "Intensity (a.u.)")
    plt.show()


def plot_gaussian_fit(file: str) -> None:
    time, intensities = extract_data(file, *MASKS[file])
    plot_v_vs_time(time, intensities, 0, save=True)



def plot_all_data(file):
    df = pd.read_csv(file)
    time_series = pd.to_numeric(df.iloc[1:, CSV_TIME_COLUMN], errors='coerce')
    intensities_series = pd.to_numeric(df.iloc[1:, CSV_VOLT_COLUMN], errors='coerce')
    plt.scatter(time_series, intensities_series, s=DATA_SIZE, label="Intensity")
    plot_config("Time (s)", "Intensity")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()
if __name__ == "__main__":
    file = file2
    plot_all_data(file)
    time, intensities = extract_data(file, *MASKS[file])
    plot_v_vs_time(time, intensities, 0)
