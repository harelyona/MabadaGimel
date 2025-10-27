import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Tuple

def gaussian(x, mu=0, sigma=1, Amplitude=1):
    """Compute the Gaussian function value for a given x, mean (mu), and standard deviation (sigma)."""
    coeff = 1 / (sigma * np.sqrt(2 * np.pi)) * Amplitude
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coeff * math.exp(exponent)

def extract_data(file_path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path)
    intensities = df.iloc[:, 4]
    time = df.iloc[:, 3]
    return time, intensities


def fit(x_data, y_data, function):
    """Fit the provided function to the data using least squares optimization."""
    params, _ = curve_fit(function, x_data, y_data)
    return params

if __name__ == "__main__":
    extract_data("data.CSV")