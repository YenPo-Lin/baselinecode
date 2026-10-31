"""Run the five baseline spectra on the current amplitude residual."""

import numpy as np

import FFT_3D
import MUSIC
from pre_processing import MA


def signal_processing(raw_csi, args):
    amplitude = np.abs(raw_csi)
    csi = amplitude - MA(amplitude, args.avg_frames)
    frame_idx = args.frame_idx

    MUSIC.ToF_Dop(args).gen_spectrum(csi, frame_idx)
    MUSIC.Azi_ToF(args).gen_spectrum(csi, frame_idx)
    MUSIC.Azi_Dop(args).gen_spectrum(csi, frame_idx)
    FFT_3D.FFT_3D(args).gen_spectrum(csi, frame_idx, fig_name="3D_FFT_sum")
    # If you want to run 3D MUSIC, you should set smaller smoothing window
    # suggest:  --stream_win 5 --freq_win 40 --freq_hop 4 --time_win 20
    # then Azi-ToF-Doppler Covariance Matrix shape = (1000, 1000)
    MUSIC.Azi_ToF_Dop(args).gen_spectrum(csi, frame_idx)

