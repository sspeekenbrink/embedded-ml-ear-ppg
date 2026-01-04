import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.interpolate import interp1d
from signal_processing.sensor_data import SensorData

from scipy.signal import find_peaks


def refine_peak_alignment(
    rpeaks_times,
    ear_signal,
    ear_time,
    window=0.2,
    search_range=0.1,
    step=0.005,
    min_hr_bpm=30,
    max_hr_bpm=240,
):
    best_shift = 0
    min_avg_distance = float("inf")
    sample_rate = 100
    min_interval = 60.0 / max_hr_bpm * sample_rate
    max_interval = 60.0 / min_hr_bpm * sample_rate
    fs_est = 1 / np.mean(np.diff(ear_time))
    min_distance_samples = int(min_interval * fs_est)
    ear_peaks_indices, _ = find_peaks(ear_signal, distance=min_distance_samples)
    ear_peaks_times = ear_time[ear_peaks_indices]

    shifts = np.arange(-search_range, search_range + step, step)

    for shift in shifts:
        shifted_rpeaks = rpeaks_times + shift
        distances = []

        for rp_time in shifted_rpeaks:
            in_window = (ear_peaks_times >= rp_time - window) & (
                ear_peaks_times <= rp_time + window
            )
            nearby_peaks = ear_peaks_times[in_window]

            if len(nearby_peaks) > 0:
                closest = nearby_peaks[np.argmin(np.abs(nearby_peaks - rp_time))]
                distances.append(abs(closest - rp_time))

        if distances:
            avg_dist = np.mean(distances)
            if avg_dist < min_avg_distance:
                min_avg_distance = avg_dist
                best_shift = shift

    return best_shift


def min_max_normalize(signal):
    """Scale an array to [0, 1] range."""
    mn, mx = np.min(signal), np.max(signal)
    return (signal - mn) / (mx - mn) if mx != mn else signal


def compute_time_offset_crosscorr(
    timeA,
    signalA,
    timeB,
    signalB,
    interp_rate=1000.0,
    round_decimals=3,
    max_offset=None,
):
    common_start = max(timeA[0], timeB[0])
    common_end = min(timeA[-1], timeB[-1])
    if common_end <= common_start:
        raise ValueError("No overlapping time range for cross-correlation.")

    num_points = int((common_end - common_start) * interp_rate)
    common_time = np.linspace(common_start, common_end, num_points)

    fA = interp1d(timeA, signalA, kind="linear", bounds_error=False, fill_value=0.0)
    fB = interp1d(timeB, signalB, kind="linear", bounds_error=False, fill_value=0.0)
    A_common = fA(common_time)
    B_common = fB(common_time)

    A_zero_mean = A_common - np.mean(A_common)
    B_zero_mean = B_common - np.mean(B_common)

    xcorr = correlate(A_zero_mean, B_zero_mean, mode="full")
    lags = np.arange(-len(A_zero_mean) + 1, len(B_zero_mean))

    if max_offset is not None:
        max_lag = int(max_offset * interp_rate)
        center = len(lags) // 2
        lag_mask = np.abs(lags) <= max_lag
        xcorr = xcorr[lag_mask]
        lags = lags[lag_mask]

    best_lag = lags[np.argmax(xcorr)]
    dt = 1.0 / interp_rate
    time_offset = best_lag * dt
    return np.round(time_offset, round_decimals)


def resample_signal(time, signal, original_rate, target_rate):
    duration = time[-1] - time[0]
    num_samples = max(2, int(duration * target_rate))
    new_time = np.linspace(time[0], time[-1], num_samples)
    interpolator = interp1d(time, signal, kind="linear", fill_value="extrapolate")
    new_signal = interpolator(new_time)
    return new_time, new_signal
