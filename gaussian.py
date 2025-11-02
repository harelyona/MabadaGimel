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
CAPSIZE = 0.5



def gaussian(x, A, B, C, D):
    return A * np.exp(-B/x - C*x)/np.sqrt(x) + D


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
    params, cov_mat = curve_fit(gaussian, time, intensities)
    x_range = np.linspace(min(time), max(time), 1000)
    fitted_curve = gaussian(x_range, *params)
    plt.plot(x_range, fitted_curve, label='Fitted Gaussian', color=FIT_COLOR)
    plt.errorbar(time, intensities, yerr=uncertainty, label='Data', color=DADA_COLOR, fmt='-o', capsize=CAPSIZE, markersize=1, elinewidth=2, ecolor=ERROR_BAR_COLOR)
    plot_config("Intensity vs Time with Gaussian Fit", "Time", "Intensity")
    if save:
        plt.savefig(f"{PLOTS_DIRECTORY}intensity_vs_time.png")
    plt.show()

def plot_config(plot_title: str, x_label: str, y_label: str) -> None:
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


if __name__ == "__main__":
    data_file = r"data\experiment1\VL_27.7_VS_43.2_d_-3.34_laser_on.csv"
    uncertainty = np.std(extract_data(r"data/experiment1/VL_27.7_VS_43.2_d_-3.34_laser_off.csv")[1])

    plot_v_vs_time(*extract_data(data_file, 0.0003, 0.0004), uncertainty)