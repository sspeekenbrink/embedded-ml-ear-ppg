from scipy.signal import butter, filtfilt
from scipy.signal import iirnotch, filtfilt


def low_pass_filter(signal, sampling_rate=99, cutoff=5, order=2):
    return _filter(signal, sampling_rate, cutoff, order, btype="low")


def high_pass_filter(signal, sampling_rate=99, cutoff=0.5, order=2):
    return _filter(signal, sampling_rate, cutoff, order, btype="high")


def _filter(signal, sampling_rate, cutoff, order, btype="low"):
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype=btype, analog=False)
    return filtfilt(b, a, signal)


def notch_filter(signal, frequency, quality, fs):
    b, a = iirnotch(frequency, quality, fs)
    return filtfilt(b, a, signal)


def bandpass_filter(signal, sampling_rate, low_cutoff, high_cutoff, order=4):
    nyquist = 0.5 * sampling_rate
    low_normalized = low_cutoff / nyquist
    high_normalized = high_cutoff / nyquist
    low_normalized = max(0.001, min(0.499, low_normalized))
    high_normalized = max(low_normalized + 0.001, min(0.499, high_normalized))
    b, a = butter(order, [low_normalized, high_normalized], btype="band", analog=False)
    return filtfilt(b, a, signal)


def filter_ecg_signal(ecg_signal, fs=250):
    filtered = notch_filter(ecg_signal, 50, 30, fs)
    filtered = notch_filter(filtered, 100, 30, fs)
    return filtered
