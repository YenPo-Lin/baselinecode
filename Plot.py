"""Plotting shared by the baseline estimators."""

import os

import matplotlib.pyplot as plt
import numpy as np


def _tof_axis_values_and_label(tau, args):
    if getattr(args, 'axis', 'ns') == 'm':
        return (tau * 300000000.0 / 2.0, 'Distance (m)')
    return (tau * 1000000000.0, 'ToF (ns)')


def _axis_values_and_label(axis_name, values, args):
    if axis_name == 'tof':
        return _tof_axis_values_and_label(values, args)
    if axis_name == 'azi':
        return (values, 'Azimuth (deg)')
    if axis_name == 'doppler':
        return (values, 'Doppler frequency (Hz)')
    raise ValueError(f'Unsupported axis name: {axis_name}')


def plot_spectrum(frame_idx, x_values, y_values, P_music, args, title='', spectrum_axes=None, sdim=None, ax=None, save=True, file_name=None):
    x_axis, y_axis = spectrum_axes
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    P_music = np.asarray(P_music)
    if P_music.shape != (len(y_values), len(x_values)):
        raise ValueError(f'Expected spectrum shape {(len(y_values), len(x_values))} for (y, x), got {P_music.shape}.')
    x_values, x_label = _axis_values_and_label(x_axis, x_values, args)
    y_values, y_label = _axis_values_and_label(y_axis, y_values, args)
    if ax is None:
        _, ax = plt.subplots()
    mesh = ax.pcolormesh(x_values, y_values, P_music, cmap='jet', shading='auto')
    colorbar = ax.figure.colorbar(mesh, ax=ax)
    colorbar.outline.set_visible(False)
    colorbar.set_label('Power (dB)', rotation=270, labelpad=15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(which='major', color='w', linestyle='-', linewidth=0.2, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    full_title = title + ' @ frame ' + str(frame_idx)
    if sdim is not None:
        full_title += ' Sdim ' + str(int(sdim))
    ax.set_title(full_title, fontsize=10)
    if save and args.pics_dir is not None:
        output_folder_by_axes = {frozenset(('azi', 'tof')): 'Azi_ToF', frozenset(('azi', 'doppler')): 'Azi_Doppler', frozenset(('tof', 'doppler')): 'ToF_Doppler'}
        save_dir = os.path.join(args.pics_dir, output_folder_by_axes[frozenset(spectrum_axes)])
        os.makedirs(save_dir, exist_ok=True)
        if file_name is None:
            file_name = f'{frame_idx:04d}.png'
        else:
            file_name = f'{frame_idx:04d}_{file_name}.png'
        save_path = os.path.join(save_dir, file_name)
        ax.figure.savefig(save_path, dpi=100)
        plt.close(ax.figure)
        print(f'Saved: {save_path}')
    return ax
