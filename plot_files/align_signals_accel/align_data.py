import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.interpolate import interp1d
from signal_processing.sensor_data import SensorData
from signal_processing.acceleration_metrics import (
    jerk_based_acceleration,
    sum_absolute_acceleration,
    euclidean_accel,
    min_max_normalize
)


def compute_time_offset_crosscorr(timeA, signalA, timeB, signalB, interp_rate=200.0):
    # Determine overlapping region
    common_start = max(timeA[0], timeB[0])
    common_end = min(timeA[-1], timeB[-1])
    if common_end <= common_start:
        raise ValueError("No overlapping time range for cross-correlation.")

    # Create common time vector
    num_points = int((common_end - common_start) * interp_rate)
    common_time = np.linspace(common_start, common_end, num_points)

    # Interpolate signals
    fA = interp1d(timeA, signalA, kind='linear')
    fB = interp1d(timeB, signalB, kind='linear')
    A_common = fA(common_time)
    B_common = fB(common_time)

    # Remove mean
    A_zero_mean = A_common - np.mean(A_common)
    B_zero_mean = B_common - np.mean(B_common)

    # Cross-correlate
    xcorr = correlate(A_zero_mean, B_zero_mean, mode='full')
    lags = np.arange(-len(A_zero_mean) + 1, len(B_zero_mean))

    # Best lag is where xcorr is maximum
    best_lag = lags[np.argmax(xcorr)]

    # Convert lag index to time offset
    dt = 1.0 / interp_rate
    time_offset = best_lag * dt
    return time_offset


if __name__ == "__main__":
    folder_path = "..."
    ear_device_signal_path = f"{folder_path}/ppg.txt"
    chest_ecg_accel_path = f"{folder_path}/acc.csv"
    ear_device_sensor_data = SensorData(file_location=ear_device_signal_path, sample_rate=100)

    # Prepare time axis for ear device - sample rate for all signals is constant
    time_ear_device = np.arange(len(ear_device_sensor_data.accel_euclidean)) / ear_device_sensor_data.sample_rate

    # Three derived accelerations from Sensor A:
    accel_ear_device = {
        'Euclidean': ear_device_sensor_data.accel_euclidean,
        'SumAbs': ear_device_sensor_data.absolute_accel,
        'Jerk': ear_device_sensor_data.jerk_accel
    }


    df_chest_accel = pd.read_csv(chest_ecg_accel_path)
    time_chest_accel = df_chest_accel['time'].values

    accel_chest_device = {
        "Euclidean": euclidean_accel(df_chest_accel['channel1'], df_chest_accel['channel2'],
                                     df_chest_accel['channel3']),
        "SumAbs": sum_absolute_acceleration(df_chest_accel['channel1'], df_chest_accel['channel2'],
                                            df_chest_accel['channel3']),
        "Jerk": jerk_based_acceleration(df_chest_accel['channel1'], df_chest_accel['channel2'],
                                        df_chest_accel['channel3'])
    }

    end_time = 60
    results = []
    for func_name, accel_ear_metric_raw in accel_ear_device.items():
        accel_chest_metric_raw = accel_chest_device[func_name]

        # Normalize
        accel_ear_metric_scaled = min_max_normalize(accel_ear_metric_raw)
        accel_chest_metric_scaled = min_max_normalize(accel_chest_metric_raw)

        # Truncate signal to end time
        time_ear_device = time_ear_device[time_ear_device <= end_time]
        accel_ear_metric_scaled = accel_ear_metric_scaled[:len(time_ear_device)]
        time_chest_accel = time_chest_accel[time_chest_accel <= end_time]
        accel_chest_metric_scaled = accel_chest_metric_scaled[:len(time_chest_accel)]

        plt.figure()
        plt.plot(time_ear_device, accel_ear_metric_scaled, label=f'A - {func_name}')
        plt.plot(time_chest_accel, accel_chest_metric_scaled, label=f'B - {func_name}')
        plt.title(f"Unaligned (Normalized): A({func_name}) vs B({func_name})")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Amplitude")
        plt.legend()
        plt.show()

        offset = compute_time_offset_crosscorr(time_ear_device, accel_ear_metric_scaled, time_chest_accel,
                                               accel_chest_metric_scaled)
        results.append((func_name, func_name, offset))

        # Shift B in time by 'offset'
        tB_aligned = time_chest_accel + offset

        plt.figure()
        plt.plot(time_ear_device, accel_ear_metric_scaled, label=f'A - {func_name}')
        plt.plot(tB_aligned, accel_chest_metric_scaled, label=f'B - {func_name} (aligned)')
        plt.title(f"Aligned (Normalized): A({func_name}) vs B({func_name})\nOffset={offset:.4f} s")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Amplitude")
        plt.legend()
        plt.show()

    print("Time Offsets (Sensor A → Sensor B) via Cross-Correlation:")
    for (func_name, chan_name, offset) in results:
        print(f"  A({func_name}) vs B({chan_name}):  {offset:.4f} s")
