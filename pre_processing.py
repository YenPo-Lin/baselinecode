"""Moving average used by the baseline input branch."""

import numpy as np


def MA(csi_amp, window_size):
    window_size = min(int(round(window_size)), len(csi_amp))
    if window_size < 1:
        raise ValueError('Moving-average window and input must be nonempty')
    window = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=0, arr=csi_amp)
