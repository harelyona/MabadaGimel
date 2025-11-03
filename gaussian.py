import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Tuple, Optional
import matplotlib.pyplot as plt

PLOTS_DIRECTORY = "plots/"
CSV_VOLT_COLUMN = 2
CSV_TIME_COLUMN = 0
FIT_COLOR = "blue"
DADA_COLOR = "black"
ERROR_BAR_COLOR = "red"
DATA_SIZE = 4
CAPSIZE = 0.5


def gaussian(x, A, mu, sigma, D):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2)) + D


def extract_data(file_path: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts time and intensity data from a CSV file.
    Optionally filters the data to a specified range
    returning the NumPy arrays.
    """
    df = pd.read_csv(file_path)
    intensities = df.iloc[1:, CSV_VOLT_COLUMN].astype(pd.Float64Dtype())
    time = df.iloc[1:, CSV_TIME_COLUMN].astype(pd.Float64Dtype())
    mask = pd.Series(True, index=time.index)
    if min_val is not None:
        mask = mask & (time > min_val)

    if max_val is not None:
        mask = mask & (time < max_val)

    time_filtered = time[mask]
    intensities_filtered = intensities[mask]
    return time_filtered.values, intensities_filtered.values



def plot_v_vs_time(time: np.ndarray, intensities: np.ndarray, uncertainty: float, save:bool = False) -> None:
    params, cov_mat = curve_fit(gaussian, time, intensities, maxfev=99999, p0=[max(intensities), time[np.argmax(intensities)], 1e-5, min(intensities)])
    x_range = np.linspace(min(time), max(time), 1000)
    fitted_curve = gaussian(x_range, *params)
    A, mu, sigma, D = params
    A_error, mu_error, sigma_error, D_error = np.sqrt(np.diag(cov_mat))
    print(f"A={A:.2e} ± {A_error:.2e}")
    print(f"mu={mu:.2e} ± {mu_error:.2e}")
    print(f"sigma={sigma:.2e} ± {sigma_error:.2e}")
    print(f"D={D:.2e} ± {D_error:.2e}")
    plt.plot(x_range, fitted_curve, label="gaussian fit", color=FIT_COLOR)
    plt.errorbar(time, intensities, yerr=uncertainty, label='Data', color=DADA_COLOR, fmt='o', capsize=CAPSIZE, markersize=1, elinewidth=2, ecolor=ERROR_BAR_COLOR)
    plot_config("Intensity vs Time with Gaussian Fit", "Time", "Intensity")
    if save:
        plt.savefig(f"{PLOTS_DIRECTORY}intensity_vs_time.png")
    plt.show()

def plot_config(plot_title: str, x_label: str, y_label: str) -> None:
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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
        time, intensities = extract_data(folder + os.sep + file_path, max_val=0.0001)
        match = re.search(r'Vs_([\d\.]+)_Vl_([\d\.]+)\.csv', file_path)
        vs_value = float(match.group(1))
        vl_value = float(match.group(2))
        slope, intercept = linear_fit(*extract_data(folder + os.sep + file_path, max_val=0), show=False)
        # intensities = intensities - (slope * time + intercept)
        plt.scatter(time, intensities, label=f"Vs={vs_value}, Vl={vl_value}", s=DATA_SIZE)
    plot_config("All Data", "Time", "Intensity")
    plt.show()


d = 0.0027
if __name__ == "__main__":
    dir = "data/Haynes-Shockley"
    #plot_all_data(dir)
    data = extract_data(f"data\Haynes-Shockley\Vs_44_Vl_27.7.csv",)

    #slope, intercept = linear_fit(*data, show=True)
    plot_v_vs_time(*data,uncertainty=0)