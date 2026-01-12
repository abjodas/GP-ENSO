from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from gp_enso.plot import plot_periodogram
from scipy.fft import fft, fftfreq

def get_dominant_period(
    y: np.ndarray,
    sample_spacing_years: float = 1.0 / 12.0,
    get_plot: bool = True,
) -> float:
    y = np.asarray(y, dtype=float)
    n = len(y)

    fft_vals = fft(y)
    fft_power = np.abs(fft_vals) ** 2

    freqs = fftfreq(n, d=sample_spacing_years)
    mask = freqs > 0

    freqs = freqs[mask]
    fft_power = fft_power[mask]

    periods = 1.0 / freqs
    peak_idx = int(np.argmax(fft_power))

    if get_plot:
        plot_periodogram(periods, fft_power)

    return float(periods[peak_idx])