"""MUSIC estimators used by the five baseline outputs."""

import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import Plot


def estimate_Sdim(Rxx, energy_ratio=0.88, min_dim=1, max_dim=None):
    arr = np.asarray(Rxx)
    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f'Rxx must be square, got {arr.shape}')
        eig_val = np.linalg.eigvalsh(arr)
    elif arr.ndim == 1:
        eig_val = arr
    else:
        raise ValueError(f'Invalid Rxx shape: {arr.shape}')
    eig_val = np.sort(np.real(eig_val))[::-1]
    eig_val = np.maximum(eig_val, 1e-12)
    M = eig_val.size
    if M < 2:
        raise ValueError('Need at least two eigenvalues.')
    if max_dim is None:
        max_dim = M - 1
    min_dim = int(np.clip(min_dim, 1, M - 1))
    max_dim = int(np.clip(max_dim, min_dim, M - 1))
    energy_ratio = float(np.clip(energy_ratio, 1e-06, 1.0))
    cumulative = np.cumsum(eig_val) / np.sum(eig_val)
    Sdim = np.searchsorted(cumulative, energy_ratio) + 1
    return int(np.clip(Sdim, min_dim, max_dim))


def resolve_Sdim(args, eig_val, label='MUSIC', sdim_attr='Sdim', energy_ratio_attr='Sdim_energy_ratio'):
    M = len(eig_val)
    if getattr(args, sdim_attr, None) is not None:
        Sdim = int(getattr(args, sdim_attr))
    else:
        energy_ratio = getattr(args, energy_ratio_attr, getattr(args, 'Sdim_energy_ratio', 0.88))
        if energy_ratio is None:
            energy_ratio = getattr(args, 'Sdim_energy_ratio', 0.88)
        Sdim = estimate_Sdim(eig_val, energy_ratio=energy_ratio, min_dim=getattr(args, 'Sdim_min', 1), max_dim=M - 1)
    Sdim = int(np.clip(Sdim, 1, M - 1))
    return Sdim


class SteeringVector:

    def __init__(self, args):
        self.args = args
        self.f_0 = args.f_0
        self.projection = args.projection
        self.antenna_spacing = args.antenna_spacing
        self.stream_win = args.stream_win
        self.freq_win = args.freq_win
        self.freq_hop = args.freq_hop
        self.time_win = args.time_win
        self.time_hop = args.time_hop
        self.time_sample_range = args.time_sample_range
        self.fs = args.fs
        self.delta_f = args.delta_f

    def steering_vector_AoA(self, theta_i, stream_win=None):
        if stream_win is None:
            stream_win = self.stream_win
        theta_i = np.deg2rad(theta_i)
        if self.projection == 'sin':
            sv = np.exp(-2j * np.pi * self.f_0 * (1 + np.sin(theta_i)) * self.antenna_spacing / 300000000.0 * np.arange(stream_win))
        elif self.projection == 'cos':
            sv = np.exp(-2j * np.pi * self.f_0 * (1 - np.cos(theta_i)) * self.antenna_spacing / 300000000.0 * np.arange(stream_win))
        else:
            raise ValueError(f'Unsupported projection: {self.projection}')
        return sv.flatten()

    def steering_vector_ToF(self, tau_i, freq_win=None, freq_hop=None):
        if freq_win is None:
            freq_win = self.freq_win
        if freq_hop is None:
            freq_hop = self.freq_hop
        row_size = freq_win // freq_hop
        sub_idx = np.arange(0, row_size) * freq_hop
        sv = np.exp(-2j * np.pi * (self.f_0 + self.delta_f * sub_idx) * tau_i)
        return sv.flatten()

    def steering_vector_Dop(self, fd, dop_win=None):
        if dop_win is None:
            dop_win = self.time_win
        t_idx = np.arange(dop_win) / self.fs
        sv = np.exp(1j * 2 * np.pi * fd * t_idx)
        return sv.flatten()

    def steering_vector_AoA_ToF(self, theta_i, tau_j, stream_win=None, freq_win=None, freq_hop=None):
        if stream_win is None:
            stream_win = self.stream_win
        if freq_win is None:
            freq_win = self.freq_win
        if freq_hop is None:
            freq_hop = self.freq_hop
        sv_azi = self.steering_vector_AoA(theta_i, stream_win)
        sv_tof = self.steering_vector_ToF(tau_j, freq_win, freq_hop)
        steering_vector = np.kron(sv_azi, sv_tof)
        return steering_vector.flatten()

    def steering_vector_ToF_Dop(self, tau, fd, freq_win=None, freq_hop=None, time_win=None):
        """
        ToF-Doppler steering vector.
        phase_tof = +2*pi*Δf*m*tau
        phase_dop = +2*pi*fd*t/fs
        sv = exp(-j*phase_tof + j*phase_dop)

        注意 flatten order = (subcarrier, time).reshape(-1)。
        """
        if freq_win is None:
            freq_win = self.freq_win
        if freq_hop is None:
            freq_hop = self.freq_hop
        if time_win is None:
            time_win = self.time_win
        freq_win = int(freq_win)
        freq_hop = int(freq_hop)
        time_win = int(time_win)
        if freq_win <= 0 or freq_hop <= 0 or time_win <= 0:
            raise ValueError('freq_win, freq_hop, and time_win must all be positive')
        freq_row_count = freq_win // freq_hop
        if freq_row_count <= 0:
            raise ValueError(f'freq_win // freq_hop must be positive, got {freq_row_count}')
        m_idx = (np.arange(freq_row_count) * freq_hop)[:, None]
        t_idx = (np.arange(time_win) / self.fs)[None, :]
        phase_tof = 2 * np.pi * self.delta_f * m_idx * tau
        phase_dop = 2 * np.pi * fd * t_idx
        sv = np.exp(-1j * (phase_tof - phase_dop))
        return sv.reshape(-1) / np.sqrt(freq_row_count * time_win)


class Azi_ToF:

    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(args, 'Azi_ToF_Sdim', getattr(args, 'Sdim', None))
        self.stream_win = int(args.stream_win)
        self.stream_sample_range = int(min(args.stream_sample_range, args.num_Rx))
        self.freq_win = int(args.freq_win)
        self.freq_hop = max(1, int(args.freq_hop))
        self.freq_space = max(1, int(args.freq_space))
        self.avg_frames = max(1, int(args.avg_frames))
        self.tau_chunk = max(1, int(getattr(args, 'azi_tof_tau_chunk', 10)))
        self.theta_grid = np.arange(args.theta_min, args.theta_max + 0.5 * args.theta_step, args.theta_step)
        self.tau_grid = np.arange(args.tau_min, args.tau_max, args.tau_step)
        self.tof_gate = float(getattr(args, 'tof_gate', 1e-08))
        if not np.isfinite(self.tof_gate) or self.tof_gate < 0:
            raise ValueError('tof_gate must be finite and non-negative')
        freq_limit = int(min(getattr(args, 'freq_sample_range', args.num_sc), args.num_sc))
        self.num_freq_samples = len(np.arange(0, freq_limit, self.freq_space))
        self.freq_win_points = self.freq_win // self.freq_hop
        if self.freq_win_points <= 0:
            raise ValueError(f'freq_win // freq_hop must be positive, got {self.freq_win_points}')
        if self.num_freq_samples <= 0:
            raise ValueError('No frequency samples are available')
        if self.stream_win <= 0:
            raise ValueError(f'stream_win must be positive, got {self.stream_win}')
        if self.stream_win > self.stream_sample_range:
            raise ValueError(f'stream_win={self.stream_win} cannot exceed stream_sample_range={self.stream_sample_range}')
        self.freq_offsets = np.arange(self.freq_win_points) * self.freq_hop
        self.freq_aperture_span = int(self.freq_offsets[-1]) + 1
        if self.freq_aperture_span > self.num_freq_samples:
            raise ValueError(f'Frequency aperture cannot exceed the available subcarriers: need {self.freq_aperture_span}, got {self.num_freq_samples}')

    def sample_csi_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        context_len = min(self.avg_frames, total_frames)
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        start = int(np.clip(frame_idx - context_len // 2, 0, total_frames - context_len))
        end = start + context_len
        return (CSI[start:end], start, end)

    def Rxx_smooth(self, CSI, frame_idx):
        csi_segment, start, end = self.sample_csi_segment(CSI, frame_idx)
        csi_segment = csi_segment[:, :, :self.stream_sample_range, :self.num_freq_samples]
        context_len, num_tx, num_rx, num_sc = csi_segment.shape
        if num_tx < 1:
            raise ValueError('Azi_ToFX requires at least one Tx')
        num_stream_slides = num_rx - self.stream_win + 1
        num_freq_slides = num_sc - int(self.freq_offsets[-1])
        sv_len = self.stream_win * self.freq_win_points
        total_snapshots = context_len * num_tx * num_stream_slides * num_freq_slides
        if total_snapshots <= 0:
            raise ValueError('Azi_ToFX smoothing produced no snapshots')
        X = np.empty((sv_len, total_snapshots), dtype=np.complex128)
        idx = 0
        for t in range(context_len):
            for tx in range(num_tx):
                for stream_start in range(num_stream_slides):
                    for freq_start in range(num_freq_slides):
                        block = csi_segment[t, tx, stream_start:stream_start + self.stream_win, :]
                        block = block[:, freq_start + self.freq_offsets]
                        X[:, idx] = block.reshape(-1)
                        idx += 1
        if idx != total_snapshots:
            raise RuntimeError(f'Snapshot mismatch: {idx} != {total_snapshots}')
        Rxx = X @ X.conj().T / total_snapshots
        Rxx = (Rxx + Rxx.conj().T) / 2.0
        '\n        print(\n            f"Azi-ToF     Rxx: {Rxx.shape}, snapshots={total_snapshots}, "\n            f"context={start}:{end}, tx={num_tx}, "\n            f"stream_slides={num_stream_slides}, "\n            f"freq_slides={num_freq_slides}"\n        )\n        '
        return Rxx

    def steering_matrix_chunk(self, tau_chunk):
        """Build unit-norm AoA-ToF steering rows for one ToF chunk."""
        sv_len = self.stream_win * self.freq_win_points
        A = np.empty((len(self.theta_grid) * len(tau_chunk), sv_len), dtype=np.complex128)
        row = 0
        normalization = np.sqrt(sv_len)
        for theta in self.theta_grid:
            for tau in tau_chunk:
                A[row] = self.steering_vector.steering_vector_AoA_ToF(theta, tau, self.stream_win, self.freq_win, self.freq_hop) / normalization
                row += 1
        return A

    def cal_spectrum(self, Rxx, sdim_attr=None, energy_ratio_attr=None, label='Azi-ToF'):
        """Calculate the linear AoA-ToF MUSIC spectrum in ToF chunks."""
        Rxx = np.asarray(Rxx, dtype=np.complex128)
        expected_dim = self.stream_win * self.freq_win_points
        if Rxx.shape != (expected_dim, expected_dim):
            raise ValueError(f'Expected Rxx shape {(expected_dim, expected_dim)}, got {Rxx.shape}')
        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val = eig_val[idx_order]
        eig_vec = eig_vec[:, idx_order]
        if sdim_attr is not None or energy_ratio_attr is not None:
            Sdim = resolve_Sdim(self.args, eig_val, label=label, sdim_attr=sdim_attr or 'Azi_ToF_Sdim', energy_ratio_attr=energy_ratio_attr or 'Sdim_energy_ratio')
        elif self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val, label=label)
        else:
            Sdim = int(np.clip(self.Sdim, 1, expected_dim - 1))
        E_s = eig_vec[:, Sdim:]
        P_azi_tof = np.empty((len(self.theta_grid), len(self.tau_grid)), dtype=float)
        for start in range(0, len(self.tau_grid), self.tau_chunk):
            end = min(start + self.tau_chunk, len(self.tau_grid))
            tau_chunk = self.tau_grid[start:end]
            SV_chunk = self.steering_matrix_chunk(tau_chunk)
            projection = SV_chunk.conj() @ E_s
            signal_projection = np.sum(np.abs(projection) ** 2, axis=1).real
            signal_projection = np.clip(signal_projection, 0.0, 1.0)
            denominator = np.maximum(np.sum(np.abs(projection) ** 2, axis=1).real, 1e-12)
            P_azi_tof[:, start:end] = (1.0 / denominator).reshape(len(self.theta_grid), len(tau_chunk))
        return (self.tau_grid, self.theta_grid, P_azi_tof, Sdim)

    def gen_spectrum(self, CSI, frame_idx, plot=True, *, sdim_attr=None, energy_ratio_attr=None, label='Azi-ToF'):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, theta_grid, P_azi_tof, Sdim = self.cal_spectrum(Rxx, sdim_attr=sdim_attr, energy_ratio_attr=energy_ratio_attr, label=label)
        if plot:
            Plot.plot_spectrum(frame_idx, x_values=theta_grid, y_values=tau_grid, P_music=10.0 * np.log10(np.maximum(P_azi_tof.T, 1e-12)), args=self.args, title='Azimuth-ToF', sdim=Sdim, spectrum_axes=('azi', 'tof'), file_name=None)
        else:
            return (tau_grid, theta_grid, P_azi_tof, Sdim)


class ToF_Dop:

    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(args, 'ToF_Dop_Sdim', getattr(args, 'Sdim', None))
        self.num_Rx = args.num_Rx
        self.freq_win = int(args.freq_win)
        self.freq_hop = max(1, args.freq_hop)
        freq_space = max(1, args.freq_space)
        freq_limit = int(min(getattr(args, 'freq_sample_range', args.num_sc), args.num_sc))
        self.num_freq_samples = len(np.arange(0, freq_limit, freq_space))
        self.time_win = int(args.time_win)
        self.time_sample_range = int(max(getattr(args, 'time_sample_range', self.time_win), self.time_win))
        self.tau_chunk = 10
        self.tau_grid = np.arange(args.tau_min, args.tau_max, args.tau_step)
        self.fd_grid = np.arange(args.doppler_min, args.doppler_max + 0.5 * args.doppler_step, args.doppler_step)
        self.epsilon = float(getattr(args, 'tof_dop_epsilon', getattr(args, 'epsilon', 1e-05)))
        freq_win_points = self.freq_win // self.freq_hop
        if freq_win_points <= 0:
            raise ValueError(f'freq_win // freq_hop must be positive, got {freq_win_points}')
        if self.num_freq_samples <= 0:
            raise ValueError('No frequency samples are available')
        freq_offsets = np.arange(freq_win_points) * self.freq_hop
        freq_aperture_span = int(freq_offsets[-1]) + 1
        if freq_aperture_span > self.num_freq_samples:
            raise ValueError(f'Frequency aperture cannot exceed the available subcarriers: need {freq_aperture_span}, got {self.num_freq_samples}')
        if self.time_win <= 0:
            raise ValueError(f'time_win must be positive, got {self.time_win}')
        if self.time_sample_range < self.time_win:
            raise ValueError(f'time_sample_range={self.time_sample_range} cannot be smaller than time_win={self.time_win}')

    def sample_csi_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        context_len = min(self.time_sample_range, total_frames)
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        start = int(np.clip(frame_idx - context_len // 2, 0, total_frames - context_len))
        end = start + context_len
        return (CSI[start:end], start, end)

    def Rxx_smooth(self, CSI, frame_idx):
        time_win = self.time_win
        freq_win = self.freq_win
        freq_hop = self.freq_hop
        freq_win_points = freq_win // freq_hop
        freq_win = np.arange(freq_win_points) * freq_hop
        csi_segment, start, end = self.sample_csi_segment(CSI, frame_idx)
        csi_segment = csi_segment[:, :, :, :self.num_freq_samples]
        context_len, num_tx, num_rx, num_sc = csi_segment.shape
        num_time_slides = context_len - time_win + 1
        num_freq_slides = num_sc - int(freq_win[-1])
        total_snapshots = num_time_slides * num_tx * num_rx * num_freq_slides
        sv_len = time_win * freq_win_points
        X = np.empty((sv_len, total_snapshots), dtype=np.complex128)
        sv_len = time_win * freq_win_points
        idx = 0
        for tx in range(num_tx):
            for rx in range(num_rx):
                for i in range(num_freq_slides):
                    for t in range(num_time_slides):
                        block = csi_segment[t:t + time_win, tx, rx, i + freq_win]
                        v = block.T.reshape(-1)
                        X[:, idx] = v
                        idx += 1
        Rxx = X @ X.conj().T / total_snapshots
        Rxx = (Rxx + Rxx.conj().T) / 2.0
        '\n        print(\n            f"ToF-Doppler Rxx: {Rxx.shape}, snapshots={total_snapshots}, "\n            f"context={start}:{end}, tx={num_tx}, "\n            f"freq_slides={num_freq_slides}, "\n            f"time_slides={num_time_slides}"\n        )\n        '
        return Rxx

    def steering_matrix_chunk(self, tau_chunk=10):
        A = np.empty((len(tau_chunk) * len(self.fd_grid), self.freq_win // self.freq_hop * self.time_win), dtype=np.complex128)
        row = 0
        for tau in tau_chunk:
            for fd in self.fd_grid:
                A[row] = self.steering_vector.steering_vector_ToF_Dop(tau, fd, self.freq_win, self.freq_hop, self.time_win)
                row += 1
        return A

    def cal_spectrum(self, Rxx, sdim_attr=None, energy_ratio_attr=None, label='ToF-Doppler'):
        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val, eig_vec = (eig_val[idx_order], eig_vec[:, idx_order])
        if sdim_attr is not None or energy_ratio_attr is not None:
            Sdim = resolve_Sdim(self.args, eig_val, label=label, sdim_attr=sdim_attr or 'ToF_Dop_Sdim', energy_ratio_attr=energy_ratio_attr or 'Sdim_energy_ratio')
        elif self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val)
        else:
            Sdim = int(np.clip(self.Sdim, 1, Rxx.shape[0] - 1))
        E_s = eig_vec[:, :Sdim]
        tau_grid = self.tau_grid
        fd_grid = self.fd_grid
        P_tof_dop = np.empty((len(tau_grid), len(fd_grid)), dtype=float)
        for start in range(0, len(tau_grid), self.tau_chunk):
            end = min(start + self.tau_chunk, len(tau_grid))
            tau_chunk = tau_grid[start:end]
            SV_chunk = self.steering_matrix_chunk(tau_chunk)
            A = SV_chunk.conj() @ E_s
            aEEa = np.real(np.sum(A * np.conj(A), axis=1))
            aEEa = np.clip(aEEa, -1.0, 1.0)
            P_tof_dop[start:end] = (1.0 / (1.0 - aEEa + self.epsilon)).reshape(len(tau_chunk), len(fd_grid))
        return (tau_grid, fd_grid, P_tof_dop, Sdim)

    def gen_spectrum(self, CSI, frame_idx, plot=True):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        tau_grid, fd_grid, P_tof_dop, Sdim = self.cal_spectrum(Rxx)
        if plot:
            Plot.plot_spectrum(frame_idx, x_values=fd_grid, y_values=tau_grid, P_music=10.0 * np.log10(np.maximum(P_tof_dop, 1e-12)), args=self.args, title='ToF-Doppler', sdim=Sdim, spectrum_axes=('doppler', 'tof'), file_name=None)
        else:
            return (tau_grid, fd_grid, P_tof_dop, Sdim)


class Azi_Dop:

    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(args, 'Azi_Dop_Sdim', getattr(args, 'Sdim', None))
        self.stream_win = int(args.stream_win)
        self.stream_sample_range = int(min(args.stream_sample_range, args.num_Rx))
        self.freq_sample_range = int(min(getattr(args, 'freq_sample_range', args.num_sc), args.num_sc))
        self.input_time_win = int(args.time_sample_range)
        self.time_win = int(args.time_win)
        self.time_hop = max(1, int(getattr(args, 'time_hop', 1)))
        self.freq_space = max(1, int(getattr(args, 'freq_space', 1)))
        self.theta = np.arange(args.theta_min, args.theta_max + 1, args.theta_step)
        self.fd_grid = np.arange(args.doppler_min, args.doppler_max + 0.5 * args.doppler_step, args.doppler_step)
        self.epsilon = float(getattr(args, 'azi_dop_epsilon', 1e-12))
        if not 0 < self.stream_win <= self.stream_sample_range:
            raise ValueError(f'Require 0 < stream_win <= stream_sample_range, got {self.stream_win} and {self.stream_sample_range}')
        if self.freq_sample_range <= 0:
            raise ValueError(f'freq_sample_range must be positive, got {self.freq_sample_range}')
        if not 0 < self.time_win < self.input_time_win:
            raise ValueError(f'Require 0 < time_win < input_time_win, got {self.time_win} and {self.input_time_win}')

    def sample_csi_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        context_len = min(self.input_time_win, total_frames)
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        start = int(np.clip(frame_idx - context_len // 2, 0, total_frames - context_len))
        end = start + context_len
        return (CSI[start:end], start, end)

    def Rxx_smooth(self, CSI, frame_idx):
        if CSI.ndim != 4:
            raise ValueError(f'Azi_DopX expects CSI shape (frame, tx, rx, subcarrier), got {CSI.shape}')
        time_win = self.time_win
        stream_win = self.stream_win
        csi_segment, start, end = self.sample_csi_segment(CSI, frame_idx)
        csi_segment = csi_segment[:, :, :self.stream_sample_range, :self.freq_sample_range]
        csi_segment = csi_segment[:, :, :, ::self.freq_space]
        csi_segment = np.asarray(csi_segment, dtype=np.complex128)
        context_len, num_tx, num_rx, num_sc = csi_segment.shape
        if context_len < self.input_time_win:
            print(f'Warning: Not enough samples for Azi-Doppler smoothing at {frame_idx}')
            return None
        time_starts = np.arange(0, context_len - time_win, self.time_hop)
        stream_starts = np.arange(0, num_rx - stream_win + 1)
        num_time_slides = len(time_starts)
        num_stream_slides = len(stream_starts)
        total_snapshots = num_tx * num_sc * num_stream_slides * num_time_slides
        sv_len = stream_win * time_win
        X = np.empty((sv_len, total_snapshots), dtype=np.complex128)
        idx = 0
        for tx in range(num_tx):
            for subc in range(num_sc):
                for stream_start in stream_starts:
                    for time_start in time_starts:
                        block = csi_segment[time_start:time_start + time_win, tx, stream_start:stream_start + stream_win, subc]
                        v = block.T.reshape(-1)
                        X[:, idx] = v
                        idx += 1
        Rxx = X @ X.conj().T / total_snapshots
        Rxx = (Rxx + Rxx.conj().T) / 2.0
        print(f'Azi-Doppler Rxx: {Rxx.shape}, snapshots={total_snapshots}, context={start}:{end}, tx={num_tx}, stream_slides={num_stream_slides}, time_slides={num_time_slides}')
        return Rxx

    def steering_matrix(self, theta, fd):
        sv_len = self.stream_win * self.time_win
        A = np.empty((len(theta) * len(fd), sv_len), dtype=np.complex128)
        row = 0
        for theta_i in theta:
            sv_azi = self.steering_vector.steering_vector_AoA(theta_i, self.stream_win)
            for fd_i in fd:
                sv_dop = self.steering_vector.steering_vector_Dop(fd_i, self.time_win)
                A[row] = np.kron(sv_azi, sv_dop) / np.sqrt(sv_len)
                row += 1
        return A

    def cal_spectrum(self, Rxx, return_sdim=False):
        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val, eig_vec = (eig_val[idx_order], eig_vec[:, idx_order])
        if self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val, label='Azi-Doppler')
        else:
            Sdim = int(np.clip(self.Sdim, 1, Rxx.shape[0] - 1))
        E_n = eig_vec[:, Sdim:]
        if E_n.size == 0:
            E_n = eig_vec[:, -1:]
        theta = self.theta
        fd = self.fd_grid
        Steering_Vectors = self.steering_matrix(theta, fd)
        A = Steering_Vectors.conj() @ E_n
        PP_flat = np.sum(np.abs(A) ** 2, axis=1)
        P_music = 10.0 * np.log10(1.0 / (PP_flat + self.epsilon))
        P_music = P_music.reshape(len(theta), len(fd))
        result = (theta, fd, P_music)
        if return_sdim:
            return (*result, Sdim)
        return result

    def gen_spectrum(self, CSI, frame_idx, x_axis='azi', y_axis='doppler'):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        if Rxx is None:
            return
        theta_grid, fd_grid, P_azi_dop, Sdim = self.cal_spectrum(Rxx, return_sdim=True)
        if (x_axis, y_axis) == ('azi', 'doppler'):
            x_values, y_values = (theta_grid, fd_grid)
            display_spectrum = P_azi_dop.T
        elif (x_axis, y_axis) == ('doppler', 'azi'):
            x_values, y_values = (fd_grid, theta_grid)
            display_spectrum = P_azi_dop
        else:
            raise ValueError("Azi_Dop axes must be ('azi', 'doppler') or reversed")
        Plot.plot_spectrum(frame_idx, x_values, y_values, display_spectrum, self.args, title='Azimuth-Doppler', sdim=Sdim, spectrum_axes=(x_axis, y_axis))


class Azi_ToF_Dop:

    def __init__(self, args):
        self.args = args
        self.steering_vector = SteeringVector(args)
        self.Sdim = getattr(args, 'Azi_ToF_Dop_Sdim', getattr(args, 'Sdim', None))
        self.stream_win = int(args.stream_win)
        self.stream_sample_range = int(min(args.stream_sample_range, args.num_Rx))
        self.freq_win = int(args.freq_win)
        self.freq_hop = max(1, int(getattr(args, 'freq_hop', 1)))
        self.freq_sample_range = int(min(getattr(args, 'freq_sample_range', args.num_sc), args.num_sc))
        self.time_win = int(args.time_win)
        self.time_hop = max(1, int(getattr(args, 'time_hop', 1)))
        self.avg_frames = args.avg_frames
        self.theta_grid = np.arange(args.theta_min, args.theta_max + 1, args.theta_step)
        self.tau_grid = np.arange(args.tau_min, args.tau_max, args.tau_step)
        self.fd_grid = np.arange(args.doppler_min, args.doppler_max + 0.5 * args.doppler_step, args.doppler_step)
        self.epsilon = float(getattr(args, 'azi_tof_dop_epsilon', 1e-12))
        freq_win_points = self.freq_win // self.freq_hop
        if not 0 < self.stream_win <= self.stream_sample_range:
            raise ValueError(f'Require 0 < stream_win <= stream_sample_range, got {self.stream_win} and {self.stream_sample_range}')
        if freq_win_points <= 0:
            raise ValueError(f'freq_win // freq_hop must be positive, got {freq_win_points}')
        freq_offsets = np.arange(freq_win_points) * self.freq_hop
        freq_aperture_span = int(freq_offsets[-1]) + 1
        if freq_aperture_span > self.freq_sample_range:
            raise ValueError(f'Frequency aperture cannot exceed the available subcarriers: need {freq_aperture_span}, got {self.freq_sample_range}')
        if self.time_win <= 0:
            raise ValueError(f'time_win must be positive, got {self.time_win}')

    def sample_csi_segment(self, CSI, frame_idx):
        total_frames = CSI.shape[0]
        frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
        if total_frames < self.time_win:
            return (CSI, 0, total_frames)
        max_rxx = 1 + (total_frames - self.time_win) // self.time_hop
        num_rxx = min(self.avg_frames, max_rxx)
        context_len = self.time_win + (num_rxx - 1) * self.time_hop
        requested_start = frame_idx - num_rxx // 2 * self.time_hop - self.time_win // 2
        start = int(np.clip(requested_start, 0, total_frames - context_len))
        end = start + context_len
        return (CSI[start:end], start, end)

    def Rxx_smooth(self, CSI, frame_idx):
        if CSI.ndim != 4:
            raise ValueError(f'Azi_ToF_Dop expects CSI shape (frame, tx, rx, subcarrier), got {CSI.shape}')
        stream_win = self.stream_win
        time_win = self.time_win
        freq_win_points = self.freq_win // self.freq_hop
        freq_offsets = np.arange(freq_win_points) * self.freq_hop
        csi_segment, start, end = self.sample_csi_segment(CSI, frame_idx)
        csi_segment = csi_segment[:, :, :self.stream_sample_range, :self.freq_sample_range]
        csi_segment = np.asarray(csi_segment, dtype=np.complex128)
        context_len, num_tx, num_rx, num_sc = csi_segment.shape
        if context_len < time_win:
            print(f'Warning: Not enough samples for 3D smoothing at {frame_idx}')
            return None
        time_starts = np.arange(0, context_len - time_win + 1, self.time_hop)
        stream_starts = np.arange(0, num_rx - stream_win + 1)
        num_time_slides = len(time_starts)
        num_stream_slides = len(stream_starts)
        num_freq_slides = num_sc - int(freq_offsets[-1])
        snapshots_per_rxx = num_tx * num_stream_slides * num_freq_slides
        total_snapshots = snapshots_per_rxx * num_time_slides
        sv_len = stream_win * freq_win_points * time_win
        Rxx = np.zeros((sv_len, sv_len), dtype=np.complex128)
        for time_start in time_starts:
            X_r = np.empty((sv_len, snapshots_per_rxx), dtype=np.complex128)
            idx = 0
            for tx in range(num_tx):
                for stream_start in stream_starts:
                    for freq_start in range(num_freq_slides):
                        subcarrier_indices = freq_start + freq_offsets
                        block = csi_segment[time_start:time_start + time_win, tx, stream_start:stream_start + stream_win, :][:, :, subcarrier_indices]
                        v = block.transpose(1, 2, 0).reshape(-1)
                        X_r[:, idx] = v
                        idx += 1
            assert idx == snapshots_per_rxx, f'Azi-ToF-Dop snapshot count mismatch: idx={idx}, expected={snapshots_per_rxx}'
            Rxx += X_r @ X_r.conj().T / snapshots_per_rxx
        Rxx /= num_time_slides
        Rxx = (Rxx + Rxx.conj().T) / 2.0
        print(f'Azi-ToF-Dop Rxx: {Rxx.shape}, averaged_Rxx={num_time_slides}, snapshots_per_Rxx={snapshots_per_rxx}, total_snapshots={total_snapshots}, context={start}:{end}, stream_slides={num_stream_slides}, subc_slides={num_freq_slides}')
        return Rxx

    def steering_matrix(self, theta_i, tau, fd):
        freq_win_points = self.freq_win // self.freq_hop
        sv_len = self.stream_win * freq_win_points * self.time_win
        A = np.empty((len(tau) * len(fd), sv_len), dtype=np.complex128)
        sv_aoa = self.steering_vector.steering_vector_AoA(theta_i, self.stream_win)
        row = 0
        for tau_i in tau:
            for fd_i in fd:
                sv_tof_dop = self.steering_vector.steering_vector_ToF_Dop(tau_i, fd_i, self.freq_win, self.freq_hop, self.time_win)
                A[row] = np.kron(sv_aoa, sv_tof_dop) / np.sqrt(self.stream_win)
                row += 1
        return A

    def cal_spectrum(self, Rxx, return_sdim=False):
        print(f'Azi-ToF-Doppler Covariance Matrix shape = {Rxx.shape}')
        eig_val, eig_vec = np.linalg.eigh(Rxx)
        idx_order = eig_val.argsort()[::-1]
        eig_val, eig_vec = (eig_val[idx_order], eig_vec[:, idx_order])
        if self.Sdim is None:
            Sdim = resolve_Sdim(self.args, eig_val, label='Azi-ToF-Doppler')
        else:
            Sdim = int(np.clip(self.Sdim, 1, Rxx.shape[0] - 1))
        E_n = eig_vec[:, Sdim:]
        if E_n.size == 0:
            E_n = eig_vec[:, -1:]
        theta = self.theta_grid
        tau = self.tau_grid
        fd = self.fd_grid
        P_music = np.zeros((len(theta), len(tau), len(fd)), dtype=float)
        for i, th in enumerate(tqdm(theta, desc='Calculating 3D MUSIC')):
            A = self.steering_matrix(th, tau, fd)
            proj = A.conj() @ E_n
            PP_flat = np.sum(np.abs(proj) ** 2, axis=1)
            P_music[i, :, :] = 10.0 * np.log10(1.0 / (PP_flat + self.epsilon)).reshape(len(tau), len(fd))
        result = (theta, tau, fd, P_music)
        if return_sdim:
            return (*result, Sdim)
        return result

    @staticmethod
    def _combine_axis(values, axis, method, axis_values=None):
        method = str(method).lower()
        if method == 'sum':
            return np.sum(values, axis=axis)
        if method == 'max':
            return np.max(values, axis=axis)
        if method == 'mean':
            return np.mean(values, axis=axis)
        if method == 'weighted':
            weights = np.abs(np.asarray(axis_values, dtype=float))
            weights = weights / (np.sum(weights) + 1e-12)
            shape = [1] * values.ndim
            shape[axis] = weights.size
            return np.sum(values * weights.reshape(shape), axis=axis)
        raise ValueError(f'Unsupported method: {method}')

    def gen_spectrum(self, CSI, frame_idx, method='sum', fig_name=None):
        Rxx = self.Rxx_smooth(CSI, frame_idx)
        theta, tau, fd, P_music_db, Sdim = self.cal_spectrum(Rxx, return_sdim=True)
        P_music = 10.0 ** (P_music_db / 10.0)
        azi_tof_db = 10.0 * np.log10(self._combine_axis(P_music, axis=2, method=method, axis_values=fd) + 1e-12)
        tof_dop_db = 10.0 * np.log10(self._combine_axis(P_music, axis=0, method=method, axis_values=theta) + 1e-12)
        azi_dop_db = 10.0 * np.log10(self._combine_axis(P_music, axis=1, method=method, axis_values=tau) + 1e-12)
        panels = [{'title': 'Azi-ToF', 'axis0_values': theta, 'axis1_values': tau, 'heatmap_db': azi_tof_db, 'data_axes': ('azi', 'tof'), 'x_axis': 'azi', 'y_axis': 'tof', 'projected_axis': 'doppler'}, {'title': 'ToF-Doppler', 'axis0_values': tau, 'axis1_values': fd, 'heatmap_db': tof_dop_db, 'data_axes': ('tof', 'doppler'), 'x_axis': 'doppler', 'y_axis': 'tof', 'projected_axis': 'azi'}, {'title': 'Azi-Doppler', 'axis0_values': theta, 'axis1_values': fd, 'heatmap_db': azi_dop_db, 'data_axes': ('azi', 'doppler'), 'x_axis': 'azi', 'y_axis': 'doppler', 'projected_axis': 'tof'}]
        fig = plt.figure(figsize=(19.2, 4.8))
        subfigures = fig.subfigures(1, 3, wspace=0)
        axes = np.asarray([subfigure.subplots() for subfigure in subfigures])
        results = []
        for ax, panel in zip(axes, panels):
            x_values = panel['axis0_values']
            y_values = panel['axis1_values']
            P_music = panel['heatmap_db'].T
            if panel['data_axes'] == (panel['y_axis'], panel['x_axis']):
                x_values, y_values = (y_values, x_values)
                P_music = panel['heatmap_db']
            Plot.plot_spectrum(frame_idx, x_values, y_values, P_music, self.args, title=panel['title'], sdim=Sdim, spectrum_axes=(panel['x_axis'], panel['y_axis']), ax=ax, save=False)
            results.append({'fig': fig, 'ax': ax, 'heatmap_db': panel['heatmap_db'], 'x_axis': panel['x_axis'], 'y_axis': panel['y_axis'], 'projected_axis': panel['projected_axis'], 'method': method})
        if self.args.pics_dir is not None:
            save_dir = os.path.join(self.args.pics_dir, 'Azi_ToF_Doppler')
            os.makedirs(save_dir, exist_ok=True)
            if fig_name is None:
                fig_name = f'3D_MUSIC_{method}'
            save_path = os.path.join(save_dir, f'{frame_idx:04d}_{fig_name}.png')
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
            print(f'Saved: {save_path}')
        return results
