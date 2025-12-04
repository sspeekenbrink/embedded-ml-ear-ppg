import scipy.signal
import heartpy as hp
import numpy as np

from typing import Tuple

class PeakFinder:
    def __init__(self, sample_rate=99, min_hr=60, max_hr=300):
        self.sample_rate = sample_rate
        self.min_hr = min_hr
        self.max_hr = max_hr

    def find_peaks_scipy(self, data):
        min_distance = int(self.sample_rate * 60 / self.max_hr)
        max_distance = int(self.sample_rate * 60 / self.min_hr)
        peaks, properties = scipy.signal.find_peaks(
            data,
            distance=min_distance,
            prominence=0.1
        )
        return peaks


    def find_peaks_heartpy(self, data):
        min_distance = int(self.sample_rate * 60 / self.max_hr)
        max_distance = int(self.sample_rate * 60 / self.min_hr)
        wd, m = hp.process(data, sample_rate=self.sample_rate, high_precision=False, clean_rr=False)
        peaks = wd['peaklist']
        return peaks


def snap_peaks_to_local_maxima(
        ppg_signal: np.ndarray,
        ppg_time: np.ndarray,
        candidate_peak_times: np.ndarray,
        window_seconds: float = 0.12
) -> Tuple[np.ndarray, np.ndarray]:
    if ppg_signal is None or ppg_time is None:
        raise ValueError("ppg_signal and ppg_time must be provided")
    if len(ppg_signal) != len(ppg_time):
        raise ValueError("ppg_signal and ppg_time must have the same length")
    if len(ppg_signal) == 0 or len(candidate_peak_times) == 0:
        return np.array([]), np.array([])

    ppg_signal = np.asarray(ppg_signal)
    ppg_time = np.asarray(ppg_time)
    candidate_peak_times = np.asarray(candidate_peak_times)

    snapped_indices = np.full(candidate_peak_times.shape, -1, dtype=int)
    snapped_times = candidate_peak_times.astype(float).copy()

    for i, t in enumerate(candidate_peak_times):
        left = t - window_seconds
        right = t + window_seconds
        mask = (ppg_time >= left) & (ppg_time <= right)
        if not np.any(mask):
            continue

        local_indices = np.where(mask)[0]
        local_signal = ppg_signal[local_indices]
        local_peaks, _ = scipy.signal.find_peaks(local_signal)

        if local_peaks.size > 0:
            peak_amplitudes = local_signal[local_peaks]
            max_amp = np.max(peak_amplitudes)
            candidates = local_peaks[peak_amplitudes == max_amp]
            if candidates.size > 1:
                candidate_times = ppg_time[local_indices[candidates]]
                best_rel = candidates[np.argmin(np.abs(candidate_times - t))]
            else:
                best_rel = candidates[0]
            chosen_global_idx = local_indices[best_rel]
        else:
            nearest_idx_in_local = np.argmin(np.abs(ppg_time[local_indices] - t))
            chosen_global_idx = local_indices[nearest_idx_in_local]

        snapped_indices[i] = chosen_global_idx
        snapped_times[i] = ppg_time[chosen_global_idx]

    return snapped_times, snapped_indices
