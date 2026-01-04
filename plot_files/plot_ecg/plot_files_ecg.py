import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import iirnotch, filtfilt, freqz

folder_path = "..."

# Get list of CSV files in folder_path
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]


def design_notch_filters(frequencies, quality, fs):
    """Returns the numerator and denominator coefficients for each requested notch filter."""
    filters = []
    for frequency in frequencies:
        b, a = iirnotch(frequency, quality, fs)
        filters.append((b, a))
    return filters


def apply_filters(signal, filters):
    """Applies each filter in sequence to the signal."""
    filtered_signal = signal
    for b, a in filters:
        filtered_signal = filtfilt(b, a, filtered_signal)
    return filtered_signal


def cascade_filter_coefficients(filters):
    """Combines multiple filters into a single equivalent transfer function."""
    b_total = np.array([1.0])
    a_total = np.array([1.0])
    for b, a in filters:
        b_total = np.convolve(b_total, b)
        a_total = np.convolve(a_total, a)
    return b_total, a_total


def process_data(file_path, filters, start=0, end=None, fs=250):
    """Loads and processes the ECG data, including filtering and FFT computation."""
    data = pd.read_csv(file_path)

    if end is None:
        end = data["time"].max()

    # Filter data based on the specified time range
    data = data[(data["time"] >= start) & (data["time"] <= end)].copy()
    if not data.empty:
        data["time"] = data["time"] - data["time"].min()

    # Sampling frequency estimation
    dt = 1 / fs

    # Normalize and enforce zero-mean on the ECG signal
    channel = data["channel"].values.astype(float)
    channel_zero_mean = channel - np.mean(channel)
    max_abs_amplitude = np.max(np.abs(channel_zero_mean))
    if max_abs_amplitude != 0:
        channel_normalized = channel_zero_mean / max_abs_amplitude
    else:
        channel_normalized = channel_zero_mean
    data["channel"] = channel_normalized

    # Apply notch filters
    data["filtered_channel"] = apply_filters(data["channel"].values, filters)

    # Compute FFT for both original and filtered signals
    signal_original = data["channel"].values
    signal_filtered = data["filtered_channel"].values
    n = len(signal_original)
    freq = np.fft.fftfreq(n, d=dt)  # Frequency axis
    fft_original = np.fft.fft(signal_original)
    fft_filtered = np.fft.fft(signal_filtered)

    fft_data = {
        "freq": freq[: n // 2],
        "original_magnitude": np.abs(fft_original[: n // 2]),
        "filtered_magnitude": np.abs(fft_filtered[: n // 2]),
    }

    return data, fft_data


def plot_data(data, fft_data, file_name):
    """Plots the ECG signal and its frequency spectrum."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot ECG Signal
    axs[0].plot(data["time"], data["channel"], label="Original ECG", color='gray', alpha=0.5)
    axs[0].plot(data["time"], data["filtered_channel"], label="Filtered ECG", color='b')
    axs[0].set_title(f"ECG Signal (50Hz Filtered)")
    axs[0].set_xlabel('Time (seconds)')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].grid()

    # Plot FFT Magnitude Spectrum
    axs[1].plot(
        fft_data["freq"],
        fft_data["filtered_magnitude"],
        label="Filtered FFT Magnitude",
        color='r'
    )
    axs[1].set_title("Frequency Spectrum of ECG Signal (50+100Hz Filtered)")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()


def plot_unfiltered_signal(data, file_name):
    """Creates a standalone figure highlighting the unfiltered ECG signal."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["time"], data["channel"], label="Original ECG", color="gray")
    ax.set_title(f"Unfiltered ECG Signal")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()


def plot_original_vs_filtered(data, file_name):
    """Creates a figure with original and filtered ECG signals stacked vertically."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(data["time"], data["channel"], color="gray", label="Original ECG")
    axs[0].set_title(f"Original ECG Signal")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(data["time"], data["filtered_channel"], color="b", label="Filtered ECG")
    axs[1].set_title(f"Filtered ECG Signal")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()


def plot_fft_original_vs_filtered(fft_data, file_name):
    """Plots the FFT of the original and filtered signals in a single figure."""
    plt.figure(figsize=(10, 4))
    plt.plot(
        fft_data["freq"],
        fft_data["original_magnitude"],
        label="Original FFT",
        color="gray",
        alpha=0.7,
    )
    plt.plot(
        fft_data["freq"],
        fft_data["filtered_magnitude"],
        label="Filtered FFT",
        color="r",
    )
    plt.title(f"FFT Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 125)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_filter_transfer_function(b_total, a_total, fs):
    """Plots the magnitude response (FFT transfer function) of the total filter."""
    w, h = freqz(b_total, a_total, worN=4096, fs=fs)

    plt.figure(figsize=(10, 4))
    plt.plot(w, np.abs(h), color="purple", label="|H(f)|")
    plt.title("Frequency Response of Combined Notch Filter (50 Hz + 100 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 125)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fs = 250
    notch_frequencies = [50, 100]
    notch_quality = 30

    filters = design_notch_filters(notch_frequencies, notch_quality, fs)
    b_total, a_total = cascade_filter_coefficients(filters)

    # Iterate through all CSV files in the folder and process + plot the data
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        data, fft_data = process_data(file_path, filters, start=40, end=60, fs=fs)
        plot_data(data, fft_data, os.path.basename(file_path))
        plot_unfiltered_signal(data, os.path.basename(file_path))
        plot_original_vs_filtered(data, os.path.basename(file_path))
        plot_fft_original_vs_filtered(fft_data, os.path.basename(file_path))

    plot_filter_transfer_function(b_total, a_total, fs)
