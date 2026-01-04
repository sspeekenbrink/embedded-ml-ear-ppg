import argparse
from pathlib import Path
import torch
import sys
import os

ECG_BASE_PATH = Path(__file__).parent / "data_files/1/ecg.csv"
EAR_DEVICE_SIGNAL_BASE_PATH = Path(__file__).parent / "data_files/1/ppg.txt"
DATA_DIR_PATH_DEFAULT = Path(sys.argv[0]).resolve().parent / "data_files"


def get_argument_parser():
    """
    Sets up the argument parser with all the configurable options for the pipeline.
    """
    p = argparse.ArgumentParser(description="PPG Peak Detection Pipeline using CNNs")

    p.add_argument(
        "--data_dir",
        type=Path,
        default=DATA_DIR_PATH_DEFAULT,
        help="Root directory containing subject data subfolders.",
    )

    p.add_argument(
        "--target_fs",
        type=float,
        default=100.0,
        help="Resample all PPG to this frequency before windowing (e.g., 100.0).",
    )

    p.add_argument(
        "--bandpass_low",
        type=float,
        default=None,
        help="Low cutoff frequency for bandpass filter (Hz). If set, both --bandpass_low and --bandpass_high must be provided.",
    )
    p.add_argument(
        "--bandpass_high",
        type=float,
        default=None,
        help="High cutoff frequency for bandpass filter (Hz). If set, both --bandpass_low and --bandpass_high must be provided.",
    )

    p.add_argument(
        "--tune_threshold",
        action="store_true",
        help="Tune decision threshold on the validation set to maximize F1 of the positive class.",
    )
    p.add_argument(
        "--decision_threshold",
        type=float,
        default=None,
        help="Use a fixed decision threshold (0..1). If set, overrides --tune_threshold.",
    )

    p.add_argument(
        "--downsample",
        type=float,
        default=None,
        help="Downsample PPG signal to this frequency before windowing.",
    )
    p.add_argument(
        "--window_s", type=float, default=4.0, help="Window size in seconds."
    )
    p.add_argument(
        "--shift_s",
        type=float,
        default=1.0,
        help="Shift between consecutive windows in seconds.",
    )

    p.add_argument(
        "--snap_peaks_window_s",
        type=float,
        default=None,
        help="If set, snap ECG-derived peaks to nearest local PPG maxima within this half-window (seconds).",
    )
    p.add_argument(
        "--ignore_peak_json",
        action="store_true",
        help="If set, ignore any peaks_snapped.json files and always recompute peaks.",
    )

    p.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs for single model training.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and evaluation.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the AdamW optimiser."
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu).",
    )
    p.add_argument(
        "--loss",
        choices=["bce", "focal"],
        default="bce",
        help="Loss function to use ('bce' or 'focal').",
    )
    p.add_argument(
        "--sample_balance",
        action="store_true",
        help="Use WeightedRandomSampler to balance windows with/without peaks.",
    )

    p.add_argument(
        "--grid_search",
        action="store_true",
        help="Run a grid search over CNN hyperparameters.",
    )
    p.add_argument(
        "--pareto_front",
        action="store_true",
        help="Run a Pareto front search over CNN hyperparameters (requires --grid_search).",
    )
    p.add_argument(
        "--epochs_per_trial",
        type=int,
        default=8,
        help="Epochs per configuration during grid search.",
    )
    p.add_argument(
        "--max_model_size",
        type=int,
        default=None,
        help="Maximum model size in Kbytes for Pareto front search.",
    )
    p.add_argument(
        "--save_top_models",
        action="store_true",
        help="Save Pareto-optimal models for later use.",
    )
    p.add_argument(
        "--top_model_dir",
        type=Path,
        default=Path("top_models"),
        help="Directory to save top models and summary.",
    )

    p.add_argument(
        "--plot_examples",
        action="store_true",
        help="Plot a few examples from the test set with model predictions.",
    )

    p.add_argument(
        "--plot_nms_comparison",
        action="store_true",
        help="Compare predictions with and without NMS. Overrides normal workflow.",
    )
    p.add_argument(
        "--focus_nms_removed",
        action="store_true",
        help="Focus on windows where NMS removes the most points (requires --plot_nms_comparison).",
    )
    p.add_argument(
        "--model_path",
        type=Path,
        default=Path("models/model_00.pt"),
        help="Path to pre-trained model for NMS comparison (default: models/model_00.pt).",
    )

    p.add_argument("--nms", type=int, default=0, help="NMS window size")

    p.add_argument(
        "--event_tol",
        type=int,
        default=None,
        help="Tolerance in samples for event-level metrics (used in event_prf1). Must be >= 0.",
    )

    p.add_argument(
        "--save_metrics_json",
        type=Path,
        default=Path("training_metrics.json"),
        help="Path to save detailed training metrics as JSON. If not provided, defaults to 'training_metrics.json'.",
    )

    return p


def parse_cli_args():
    """
    Parses command-line arguments and performs initial validation.
    """
    parser = get_argument_parser()
    args = parser.parse_args()

    if args.pareto_front and not args.grid_search:
        parser.error("--pareto_front requires --grid_search to be enabled.")

    if args.max_model_size and not args.pareto_front:
        parser.error("--max_model_size is only relevant for Pareto front search.")

    if args.save_top_models and not args.pareto_front:
        parser.error("--save_top_models is only relevant for Pareto front search.")

    if args.target_fs <= 0:
        parser.error("--target_fs must be > 0")

    if args.downsample is not None and args.downsample <= 0:
        parser.error("--downsample must be > 0")

    if args.downsample is not None and args.downsample >= args.target_fs:
        print(
            "[Warn] --downsample >= --target_fs; final rate will be the same or higher than the common rate."
        )

    if args.decision_threshold is not None:
        args.decision_threshold = float(
            max(1e-6, min(1.0 - 1e-6, args.decision_threshold))
        )

    if args.event_tol is not None and args.event_tol < 0:
        parser.error("--event_tol must be >= 0")

    if (args.bandpass_low is not None) != (args.bandpass_high is not None):
        parser.error(
            "Both --bandpass_low and --bandpass_high must be provided together."
        )

    if args.bandpass_low is not None and args.bandpass_high is not None:
        if args.bandpass_low <= 0:
            parser.error("--bandpass_low must be > 0")
        if args.bandpass_high <= 0:
            parser.error("--bandpass_high must be > 0")
        if args.bandpass_low >= args.bandpass_high:
            parser.error("--bandpass_low must be < --bandpass_high")
        if args.bandpass_high >= args.target_fs / 2:
            parser.error("--bandpass_high must be < target_fs/2 (Nyquist frequency)")

    return args
