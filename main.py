import argparse
from pathlib import Path

import numpy as np

from signal_processing import signal_processing


def sdim_or_auto(value):
    if value.lower() == "auto":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Sdim must be an integer or 'auto'") from exc


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csi_file",
        type=Path,
        default=Path(
            "/Users/YPL/Documents/Experiments/CSI_files_2026/intermediates/"
            "20261030-180158_move1/merged_csi/20261030-180158_move1_merged.npz"
        ),
    )
    parser.add_argument("--pics_dir", type=Path)

    # Plotted frame index
    parser.add_argument("--frame_idx", type=int, default=666)

    # Hardware Settings
    parser.add_argument("--f_0", type=float, default=5.57e9)
    parser.add_argument("--fs", type=float, default=100)
    parser.add_argument("--antenna_spacing", type=float, default=0.02)
    parser.add_argument("--projection", choices=("sin", "cos"), default="sin")
    # if choose "cos", theta range should be (0~180)

    # Correlation matrix time average snapshots
    parser.add_argument("--avg_frames", type=int, default=100)

    # MUSIC smoothing parameters
    parser.add_argument("--stream_win", type=int, default=5)
    parser.add_argument("--stream_sample_range", type=int, default=8)
    parser.add_argument("--freq_win", type=int, default=56)
    parser.add_argument("--freq_hop", type=int, default=2)
    parser.add_argument("--freq_sample_range", type=int, default=64)
    parser.add_argument("--freq_space", type=int, default=1)
    parser.add_argument("--time_win", type=int, default=30)
    parser.add_argument("--time_hop", type=int, default=1)
    parser.add_argument("--time_sample_range", type=int, default=50)

    # Heatmap range and step settings
    parser.add_argument("--theta_min", type=float, default=-30)
    parser.add_argument("--theta_max", type=float, default=31)
    parser.add_argument("--theta_step", type=float, default=2)
    parser.add_argument("--tau_min", type=float, default=4e-9)
    parser.add_argument("--tau_max", type=float, default=22e-9)
    parser.add_argument("--tau_step", type=float, default=3e-10)
    parser.add_argument("--doppler_min", type=float, default=-25)
    parser.add_argument("--doppler_max", type=float, default=25)
    parser.add_argument("--doppler_step", type=float, default=0.5)

    # MUSIC signal dimension: constant/ dynamic
    parser.add_argument("--Azi_ToF_Sdim", type=int, default=20,)
    parser.add_argument("--ToF_Dop_Sdim", type=int, default=None)
    parser.add_argument("--Azi_Dop_Sdim", type=int, default=None)
    parser.add_argument("--Azi_ToF_Dop_Sdim", type=int)
    parser.add_argument("--Sdim_energy_ratio", type=float, default=0.33)
    parser.add_argument("--axis", choices=("m", "ns"), default="m")
    return parser


def main():
    args = create_parser().parse_args()
    with np.load(args.csi_file, allow_pickle=False) as data:
        csi = data["csi"]
        if csi.ndim != 4:
            raise ValueError("CSI must have shape (frame, Tx, Rx, subcarrier)")
        args.num_frames, args.num_Tx, args.num_Rx, args.num_sc = csi.shape
        if "frequency_offsets_hz" in data:
            args.frequency_offsets_hz = np.asarray(data["frequency_offsets_hz"], dtype=float)
        if "frequency_spacing_hz" in data:
            args.delta_f = float(data["frequency_spacing_hz"].item())
        elif hasattr(args, "frequency_offsets_hz"):
            args.delta_f = float(np.median(np.diff(args.frequency_offsets_hz)))
        else:
            raise ValueError("NPZ needs frequency_spacing_hz or frequency_offsets_hz")

    if args.pics_dir is None:
        args.pics_dir = Path(__file__).resolve().parent / args.csi_file.stem
    if not 0 <= args.frame_idx < args.num_frames:
        raise ValueError("frame_idx is outside CSI")
    signal_processing(csi, args)


if __name__ == "__main__":
    main()
