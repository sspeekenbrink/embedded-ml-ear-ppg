import json
from dataclasses import dataclass
from pathlib import Path

from fractions import Fraction
from scipy.signal import resample_poly

import numpy as np
import pandas as pd
import torch
import logging
from torch.utils.data import Dataset
from scipy.signal import decimate
from typing import List, Tuple
import argparse

from wfdb.processing import xqrs_detect
from signal_processing.filter import filter_ecg_signal, bandpass_filter
from signal_processing.align_data import min_max_normalize
from signal_processing.sensor_data import SensorData
from signal_processing.find_peaks import snap_peaks_to_local_maxima

DEFAULT_ECG_FS = 250.0


@dataclass(frozen=True)
class DataSource:
    """
    Represents a single, complete data recording session with all necessary
    file paths and metadata.
    """

    id: str
    ppg_path: Path
    ecg_path: Path
    ppg_fs: float
    ecg_fs: float
    ecg_offset: float
    exclude_ranges: List[dict]  # List of {"start": int, "end": int} dictionaries


def find_data_sources(root_dir: Path) -> List[DataSource]:
    """
    Scans a root directory for subject subfolders, each expected to contain
    ppg.txt, ecg.csv, and data.json.

    Args:
        root_dir: The path to the directory containing all the data subfolders.

    Returns:
        A list of DataSource objects, one for each valid subject folder found.
    """
    if not root_dir.is_dir():
        logging.error(f"Data directory not found: {root_dir}")
        return []

    sources = []
    for subject_dir in root_dir.iterdir():
        if not subject_dir.is_dir():
            continue

        ppg_file = subject_dir / "ppg.txt"
        ecg_file = subject_dir / "ecg.csv"
        json_file = subject_dir / "data.json"

        if not all([ppg_file.exists(), ecg_file.exists(), json_file.exists()]):
            logging.warning(
                f"Skipping directory '{subject_dir.name}': missing one or more required files "
                "(ppg.txt, ecg.csv, data.json)."
            )
            continue

        try:
            with open(json_file, "r") as f:
                metadata = json.load(f)

            exclude_ranges = metadata.get("exclude_ranges", [])

            source = DataSource(
                id=subject_dir.name,
                ppg_path=ppg_file,
                ecg_path=ecg_file,
                ppg_fs=float(metadata["ppg_fs"]),
                ecg_fs=float(metadata.get("ecg_fs", DEFAULT_ECG_FS)),
                ecg_offset=float(metadata["ecg_offset"]),
                exclude_ranges=exclude_ranges,
            )
            sources.append(source)
            logging.info(
                f"Successfully found and parsed data source: '{subject_dir.name}'"
            )
        except (KeyError, json.JSONDecodeError, TypeError) as e:
            logging.error(f"Failed to parse metadata for '{subject_dir.name}': {e}")

    return sources


def downsample_ppg(
    ppg_signal: np.ndarray, original_fs: float, target_fs: float
) -> Tuple[np.ndarray, float]:
    """
    Change the sampling rate of a PPG signal to target_fs using polyphase filtering.
    Works for arbitrary ratios (not just integer factors). Keeps the name for backward compatibility.
    """
    if target_fs <= 0:
        raise ValueError("target_fs must be > 0")

    resampled, new_fs = resample_to_rate(ppg_signal, original_fs, target_fs)
    logging.info(
        f"Resampled PPG from {original_fs:.6f} Hz to {new_fs:.6f} Hz (len {len(ppg_signal)} → {len(resampled)})"
    )
    return resampled, new_fs


def map_rpeaks_to_ppg_indices(
    ecg_signal: np.ndarray,
    ecg_times: np.ndarray,
    ecg_fs: float,
    ppg_fs: float,
    ecg_offset_s: float,
) -> np.ndarray:
    """
    Detects R-peaks in an ECG signal and maps their timestamps to indices in the PPG signal.
    """
    ecg_filtered = filter_ecg_signal(ecg_signal, fs=ecg_fs)
    ecg_norm = min_max_normalize(ecg_filtered)
    rpeaks = xqrs_detect(ecg_norm, fs=ecg_fs, verbose=False)

    rpeak_times = ecg_times[rpeaks] + ecg_offset_s
    return np.round(rpeak_times * ppg_fs).astype(int)


def _load_snapped_peaks_if_available(
    subject_dir: Path, *, ignore_peak_json: bool
) -> np.ndarray | None:
    """
    Loads precomputed snapped peaks times (seconds) if a peaks_snapped.json exists.
    Returns None if not present or when ignore_peak_json is True.
    """
    if ignore_peak_json:
        return None
    jpath = subject_dir / "peaks_snapped.json"
    if not jpath.exists():
        return None
    with open(jpath, "r") as f:
        data = json.load(f)
    times = data.get("times_s")
    if not isinstance(times, list):
        return None
    arr = np.asarray(times, dtype=float)
    if arr.ndim != 1:
        return None
    return arr


def create_ppg_windows(
    ppg: np.ndarray,
    rpeak_indices: np.ndarray,
    fs: float,
    window_s: float,
    shift_s: float,
    exclude_ranges: List[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates windows from a PPG signal and generates corresponding binary labels for peaks.
    Each window is independently min-max normalized.
    Excludes windows that overlap with the specified exclude_ranges.
    """
    win_len = int(round(window_s * fs))
    step = int(round(shift_s * fs))

    rpeak_indices = rpeak_indices[(rpeak_indices >= 0) & (rpeak_indices < len(ppg))]
    if len(rpeak_indices) < 2:
        return np.array([]), np.array([])

    start_idx, end_idx = rpeak_indices[0], rpeak_indices[-1]
    ppg_trimmed = ppg[start_idx : end_idx + 1]
    rpeaks_relative = rpeak_indices - start_idx

    labels_full = np.zeros(len(ppg_trimmed), dtype=np.int64)
    labels_full[rpeaks_relative] = 1

    exclude_ranges = exclude_ranges or []
    relative_exclude_ranges = []
    for range_dict in exclude_ranges:
        exclude_start = range_dict["start"] - start_idx
        exclude_end = range_dict["end"] - start_idx
        # Only include ranges that overlap with the trimmed signal
        if exclude_end >= 0 and exclude_start < len(ppg_trimmed):
            relative_exclude_ranges.append(
                {
                    "start": max(0, exclude_start),
                    "end": min(len(ppg_trimmed), exclude_end),
                }
            )

    def window_overlaps_excluded_range(window_start: int, window_end: int) -> bool:
        """Check if a window overlaps with any excluded range."""
        for exclude_range in relative_exclude_ranges:
            if (
                window_start < exclude_range["end"]
                and window_end > exclude_range["start"]
            ):
                return True
        return False

    xs, ys = [], []
    for beg in range(0, len(ppg_trimmed) - win_len + 1, step):
        window_end = beg + win_len

        if window_overlaps_excluded_range(beg, window_end):
            continue

        ppg_window = ppg_trimmed[beg : beg + win_len]
        label_window = labels_full[beg : beg + win_len]

        w_min, w_max = np.min(ppg_window), np.max(ppg_window)
        denominator = w_max - w_min
        if denominator > 1e-6:
            normalized_window = (ppg_window - w_min) / denominator
        else:
            normalized_window = np.zeros_like(ppg_window)

        xs.append(normalized_window)
        ys.append(label_window)

    if not xs:
        return np.array([]), np.array([])

    return np.stack(xs), np.stack(ys)


def resample_to_rate(
    signal: np.ndarray,
    original_fs: float,
    target_fs: float,
    *,
    max_denominator: int = 1000,
) -> Tuple[np.ndarray, float]:
    """
    Resample 'signal' from original_fs to target_fs using polyphase filtering.
    Works for slight up/down sampling (e.g., ~99–100 Hz -> 100 Hz).
    Returns (resampled_signal, target_fs).
    """
    if target_fs <= 0 or original_fs <= 0:
        raise ValueError("Both original_fs and target_fs must be > 0")

    if np.isclose(original_fs, target_fs, rtol=0.0, atol=1e-9):
        return signal, float(original_fs)

    ratio = Fraction(target_fs / original_fs).limit_denominator(max_denominator)
    up, down = ratio.numerator, ratio.denominator

    y = resample_poly(signal, up, down)

    expected_len = int(round(len(signal) * target_fs / original_fs))
    if len(y) != expected_len:
        if len(y) > expected_len:
            y = y[:expected_len]
        else:
            y = np.pad(y, (0, expected_len - len(y)), mode="edge")

    return y.astype(signal.dtype, copy=False), float(target_fs)


def create_dataset_from_sources(
    sources: List[DataSource], args: "argparse.Namespace"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes all data sources, creates windows, and concatenates them into a single dataset.
    """
    all_windows, all_labels = [], []

    target_fs = float(getattr(args, "target_fs", 100.0))
    logging.info(
        f"[Dataset] target_fs={target_fs:.6f} Hz"
        + (f", final downsample={args.downsample:.6f} Hz" if args.downsample else "")
    )

    for source in sources:
        logging.info(
            f"Processing source: {source.id} (ppg_fs={source.ppg_fs:.6f}, ecg_fs={source.ecg_fs:.6f})"
        )

        # Load raw
        ppg_raw = SensorData(
            file_location=str(source.ppg_path), sample_rate=source.ppg_fs
        ).led_green_values
        df_ecg = pd.read_csv(source.ecg_path)
        ecg_raw, ecg_t = df_ecg["channel"].values, df_ecg["time"].values

        current_ppg, current_ppg_fs = resample_to_rate(
            ppg_raw, source.ppg_fs, target_fs
        )

        if args.downsample:
            current_ppg, current_ppg_fs = downsample_ppg(
                current_ppg, current_ppg_fs, float(args.downsample)
            )

        if (
            hasattr(args, "bandpass_low")
            and hasattr(args, "bandpass_high")
            and args.bandpass_low is not None
            and args.bandpass_high is not None
        ):
            logging.info(
                f"{source.id}: applying bandpass filter ({args.bandpass_low:.2f} - {args.bandpass_high:.2f} Hz) to PPG signal"
            )
            current_ppg = bandpass_filter(
                current_ppg, current_ppg_fs, args.bandpass_low, args.bandpass_high
            )

        snapped_times_s = _load_snapped_peaks_if_available(
            source.ppg_path.parent,
            ignore_peak_json=bool(getattr(args, "ignore_peak_json", False)),
        )

        if snapped_times_s is not None:
            rpeaks_idx = np.round(snapped_times_s * current_ppg_fs).astype(int)
            logging.info(
                f"{source.id}: using {len(rpeaks_idx)} peaks from peaks_snapped.json"
            )
        else:
            rpeaks_idx = map_rpeaks_to_ppg_indices(
                ecg_raw, ecg_t, source.ecg_fs, current_ppg_fs, source.ecg_offset
            )

            snap_w = getattr(args, "snap_peaks_window_s", None)
            if snap_w is not None:
                ppg_time = np.arange(len(current_ppg)) / float(current_ppg_fs)
                candidate_times = rpeaks_idx / float(current_ppg_fs)
                snapped_times, snapped_idx = snap_peaks_to_local_maxima(
                    ppg_signal=current_ppg,
                    ppg_time=ppg_time,
                    candidate_peak_times=candidate_times,
                    window_seconds=float(snap_w),
                )
                valid = snapped_idx >= 0
                rpeaks_idx = snapped_idx[valid]
                logging.info(
                    f"{source.id}: snapped peaks within ±{float(snap_w):.3f}s window -> {len(rpeaks_idx)} peaks"
                )

        if source.exclude_ranges:
            logging.info(
                f"{source.id}: excluding {len(source.exclude_ranges)} time ranges from windowing"
            )

            windows, labels = create_ppg_windows(
                current_ppg,
                rpeaks_idx,
                current_ppg_fs,
                args.window_s,
                args.shift_s,
                source.exclude_ranges,
            )

        expected_len = int(round(args.window_s * current_ppg_fs))
        if windows.size > 0 and windows.shape[1] != expected_len:
            wl = windows.shape[1]
            if wl > expected_len:
                windows = windows[:, :expected_len]
                labels = labels[:, :expected_len]
            else:
                pad = expected_len - wl
                windows = np.pad(windows, ((0, 0), (0, pad)), mode="edge")
                labels = np.pad(labels, ((0, 0), (0, pad)), mode="edge")
            logging.warning(
                f"[Dataset] {source.id}: adjusted window length {wl}→{expected_len}"
            )

        if windows.size > 0:
            all_windows.append(windows.astype(np.float32, copy=False))
            all_labels.append(labels.astype(np.int64, copy=False))
            logging.info(
                f"-> {source.id}: {len(windows)} windows, shape {windows.shape} at {current_ppg_fs:.6f} Hz"
            )
        else:
            logging.warning(
                f"-> No windows from {source.id}. Check signal quality/alignment."
            )

    if not all_windows:
        logging.error("No windows were generated from any data source. Cannot proceed.")
        return np.array([]), np.array([])

    final_windows = np.concatenate(all_windows, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    logging.info(
        f"[Dataset] Final windows {final_windows.shape}, labels {final_labels.shape}"
    )

    return final_windows, final_labels


class PPGWindowDataset(Dataset):
    """PyTorch Dataset for PPG windows."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Input arrays x and y must be 2-dimensional.")
        self.x = torch.from_numpy(x).float().unsqueeze(1)
        self.y = torch.from_numpy(y).float().unsqueeze(1)

        self.has_peak = self.y.view(self.y.size(0), -1).sum(dim=1) > 0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
