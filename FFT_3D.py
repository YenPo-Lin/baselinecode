"""3D FFT estimator used by the baseline output."""

import os

import matplotlib.pyplot as plt
import numpy as np

import Plot


class FFT_3D:

    def __init__(self, args):
        self.args = args
        self.theta_grid = np.arange(args.theta_min, args.theta_max + 1, args.theta_step, dtype=float)
        self.tau_grid = np.arange(args.tau_min, args.tau_max, args.tau_step, dtype=float)
        self.fd_grid = np.arange(args.doppler_min, args.doppler_max + 0.5 * args.doppler_step, args.doppler_step, dtype=float)
        self.time_win = int(args.time_win)
        self.time_hop = int(getattr(args, 'time_hop', 1))
        self.avg_frames = int(args.avg_frames)
        self.last_cube = None
        self.last_snapshot_count = 0
        self.last_context = None
        if any((grid.size == 0 for grid in (self.theta_grid, self.tau_grid, self.fd_grid))):
            raise ValueError('Azimuth, ToF, and Doppler scan grids must be nonempty')
        if self.time_win <= 1 or self.time_hop <= 0 or self.avg_frames <= 0:
            raise ValueError('time_win must exceed 1; time_hop and avg_frames must be positive')
        if not np.isfinite(args.fs) or args.fs <= 0:
            raise ValueError('fs must be positive and finite')
        if not np.isfinite(args.antenna_spacing) or args.antenna_spacing <= 0:
            raise ValueError('antenna_spacing must be positive and finite')
        if args.projection not in ('sin', 'cos'):
            raise ValueError("projection must be 'sin' or 'cos'")

    def _frequency_offsets(self, num_sc):
        offsets = getattr(self.args, 'frequency_offsets_hz', None)
        if offsets is None:
            spacing = getattr(self.args, 'delta_f', None)
            if spacing is None or not np.isfinite(spacing) or spacing <= 0:
                raise ValueError('Provide frequency_offsets_hz or positive delta_f in args')
            offsets = np.arange(num_sc, dtype=float) * float(spacing)
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (num_sc,) or not np.all(np.isfinite(offsets)):
            raise ValueError('frequency_offsets_hz must match the CSI subcarrier axis')
        return offsets - offsets[0]

    def _window_starts(self, total_frames, frame_idx):
        if not 0 <= frame_idx < total_frames:
            raise ValueError('frame_idx is outside CSI')
        if total_frames < self.time_win:
            raise ValueError('CSI has fewer frames than time_win')
        max_snapshots = 1 + (total_frames - self.time_win) // self.time_hop
        count = min(self.avg_frames, max_snapshots)
        context_len = self.time_win + (count - 1) * self.time_hop
        requested_start = frame_idx - count // 2 * self.time_hop - self.time_win // 2
        start = int(np.clip(requested_start, 0, total_frames - context_len))
        return (start + np.arange(count) * self.time_hop, start, start + context_len)

    def cal_spectrum(self, CSI, frame_idx, *, tx_index=None):
        """Return ``(theta, tau, fd, cube_linear)`` with shape ``(azi,tof,dop)``.

        ``CSI`` is a preprocessed real amplitude array with dimensions
        ``(frame, Tx, Rx, subcarrier)``.  The caller selects its BG or MA
        preprocessing branch; this method performs no further preprocessing.
        """
        csi = np.asarray(CSI)
        if csi.ndim != 4 or min(csi.shape) == 0:
            raise ValueError('CSI must have shape (frame, Tx, Rx, subcarrier)')
        if np.iscomplexobj(csi):
            raise ValueError('FFT_3D expects preprocessed real CSI amplitude, not complex CSI')
        if not np.all(np.isfinite(csi)):
            raise ValueError('CSI must contain only finite values')
        total_frames, num_tx, num_rx, num_sc = csi.shape
        if tx_index is None:
            tx_indices = range(num_tx)
        else:
            tx_index = int(tx_index)
            if not 0 <= tx_index < num_tx:
                raise ValueError('tx_index is outside the CSI Tx axis')
            tx_indices = (tx_index,)
        starts, context_start, context_end = self._window_starts(total_frames, int(frame_idx))
        offsets = self._frequency_offsets(num_sc)
        if self.args.projection == 'sin':
            spatial_coordinate = 1.0 + np.sin(np.deg2rad(self.theta_grid))
        else:
            spatial_coordinate = 1.0 - np.cos(np.deg2rad(self.theta_grid))
        rx_index = np.arange(num_rx, dtype=float)
        spatial_kernel = np.exp(2j * np.pi * float(self.args.f_0) * float(self.args.antenna_spacing) / 300000000.0 * rx_index[:, None] * spatial_coordinate[None, :])
        delay_kernel = np.exp(2j * np.pi * offsets[:, None] * self.tau_grid[None, :])
        slow_time_s = np.arange(self.time_win, dtype=float) / float(self.args.fs)
        doppler_kernel = np.exp(-2j * np.pi * slow_time_s[:, None] * self.fd_grid[None, :])
        cube = np.zeros((self.theta_grid.size, self.tau_grid.size, self.fd_grid.size), dtype=float)
        normalizer = float(num_rx * num_sc * self.time_win)
        for tx in tx_indices:
            for start in starts:
                block = csi[start:start + self.time_win, tx].transpose(1, 2, 0)
                time_projection = np.einsum('rqt,tf->rqf', block, doppler_kernel, optimize=True)
                delay_projection = np.einsum('rqf,qd->rdf', time_projection, delay_kernel, optimize=True)
                spectrum = np.einsum('ra,rdf->adf', spatial_kernel, delay_projection, optimize=True)
                cube += np.abs(spectrum / normalizer) ** 2
        cube /= len(tx_indices) * len(starts)
        self.last_cube = cube
        self.last_snapshot_count = len(starts)
        self.last_context = (context_start, context_end)
        return (self.theta_grid, self.tau_grid, self.fd_grid, cube)

    @staticmethod
    def _project(cube, axis, method, axis_values):
        method = method.lower()
        if method == 'sum':
            return np.sum(cube, axis=axis)
        if method == 'max':
            return np.max(cube, axis=axis)
        if method == 'mean':
            return np.mean(cube, axis=axis)
        if method == 'weighted':
            weights = np.abs(np.asarray(axis_values, dtype=float))
            weights /= np.sum(weights) + 1e-12
            shape = [1] * cube.ndim
            shape[axis] = weights.size
            return np.sum(cube * weights.reshape(shape), axis=axis)
        raise ValueError("method must be 'sum', 'max', 'mean', or 'weighted'")

    def gen_spectrum(self, CSI, frame_idx, method='sum', fig_name=None, *, tx_index=None):
        """Save the same Azi-ToF, ToF-Doppler, Azi-Doppler panel layout as 3D MUSIC."""
        theta, tau, fd, cube = self.cal_spectrum(CSI, frame_idx, tx_index=tx_index)
        floor = 1e-12
        panels = (('Azi-ToF', theta, tau, 'azi', 'tof', 'doppler', self._project(cube, 2, method, fd).T), ('ToF-Doppler', fd, tau, 'doppler', 'tof', 'azi', self._project(cube, 0, method, theta)), ('Azi-Doppler', theta, fd, 'azi', 'doppler', 'tof', self._project(cube, 1, method, tau).T))
        fig = plt.figure(figsize=(19.2, 4.8))
        subfigures = fig.subfigures(1, 3, wspace=0)
        axes = [subfigure.subplots() for subfigure in subfigures]
        results = []
        for ax, (title, x_values, y_values, x_axis, y_axis, projected_axis, linear_map) in zip(axes, panels):
            heatmap_db = 10.0 * np.log10(np.maximum(linear_map, floor))
            Plot.plot_spectrum(frame_idx, x_values, y_values, heatmap_db, self.args, title=title, spectrum_axes=(x_axis, y_axis), ax=ax, save=False)
            results.append({'fig': fig, 'ax': ax, 'heatmap_db': heatmap_db, 'x_axis': x_axis, 'y_axis': y_axis, 'projected_axis': projected_axis, 'method': method})
        if self.args.pics_dir is not None:
            save_dir = os.path.join(self.args.pics_dir, 'Azi_ToF_Doppler')
            os.makedirs(save_dir, exist_ok=True)
            if fig_name is None:
                fig_name = f'3D_FFT_{method}'
            save_path = os.path.join(save_dir, f'{int(frame_idx):04d}_{fig_name}.png')
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
            print(f'Saved: {save_path}')
        return results
