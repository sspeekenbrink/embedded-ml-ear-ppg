import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import logging

from .find_peaks import PeakFinder
import pandas as pd
from .filter import low_pass_filter, high_pass_filter, bandpass_filter

from .acceleration_metrics import euclidean_accel, jerk_based_acceleration, sum_absolute_acceleration


class SensorData:
    def __init__(self, file_location, sample_rate=99):
        self.sample_rate = sample_rate
        self.file_location = file_location
        self.led_green_values = None
        self.led_ir_values = None
        self.led_red_values = None
        self.ambient_values = None
        self.acceleration_x = None
        self.acceleration_y = None
        self.acceleration_z = None

        self.accel_euclidean = None
        self.absolute_accel = None
        self.jerk_accel = None
        self.rms_accel = None

        self.format_file()

    def filter_signal(self, signal):
        new_signal = low_pass_filter(signal, self.sample_rate, 8, 4)
        new_signal = high_pass_filter(new_signal, self.sample_rate, 0.5, 4)
        return new_signal

    def filter_all_ppg_signals(self):
        self.led_green_values = self.filter_signal(self.led_green_values)
        self.led_ir_values = self.filter_signal(self.led_ir_values)
        self.led_red_values = self.filter_signal(self.led_red_values)

    def format_file(self):
        data_list = []
        with open(self.file_location, "r") as file:
            for line in file:
                try:
                    result = [int(value.strip()) for value in line.split(",")]
                except ValueError:
                    if line.startswith("Command"):
                        continue
                    else:
                        raise ValueError("Invalid line: " + line)
                if len(result) == 7:
                    data_list.append(result)

        data_array = np.array(data_list, dtype=np.int32)
        self.led_green_values = data_array[:, 0]
        self.led_ir_values = data_array[:, 1]
        self.led_red_values = data_array[:, 2]
        self.ambient_values = data_array[:, 3]
        self.acceleration_x = data_array[:, 4]
        self.acceleration_y = data_array[:, 5]
        self.acceleration_z = data_array[:, 6]
        self.scale_acceleration()
        self.accel_euclidean = self.euclidean_accel()
        self.absolute_accel = self.sum_absolute_acceleration()
        self.jerk_accel = self.jerk_based_acceleration()
        self.rms_accel = self.moving_rms_acceleration()

        logging.debug("Data formatted successfully.")

    @staticmethod
    def normalize_data(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def scale_acceleration(self):
        self.acceleration_x = self.acceleration_x / 16348.0
        self.acceleration_y = self.acceleration_y / 16348.0
        self.acceleration_z = self.acceleration_z / 16348.0

    def euclidean_accel(self):
        return euclidean_accel(self.acceleration_x, self.acceleration_y, self.acceleration_z)

    def sum_absolute_acceleration(self):
        return sum_absolute_acceleration(self.acceleration_x, self.acceleration_y, self.acceleration_z)

    def jerk_based_acceleration(self):
        return jerk_based_acceleration(
            self.acceleration_x, self.acceleration_y, self.acceleration_z, self.sample_rate
        )

    def moving_rms_acceleration(self, window_size=5):
        a_abs = np.sqrt(
            self.acceleration_x**2 + self.acceleration_y**2 + self.acceleration_z**2
        )
        kernel = np.ones(window_size) / window_size
        return np.convolve(a_abs**2, kernel, mode="same") ** 0.5

    def cutoff_index(self, start_index, end_index):
        self.led_green_values = self.led_green_values[start_index:end_index]
        self.led_ir_values = self.led_ir_values[start_index:end_index]
        self.led_red_values = self.led_red_values[start_index:end_index]
        self.ambient_values = self.ambient_values[start_index:end_index]
        self.acceleration_x = self.acceleration_x[start_index:end_index]
        self.acceleration_y = self.acceleration_y[start_index:end_index]
        self.acceleration_z = self.acceleration_z[start_index:end_index]

    def cutoff_time(self, start_time, end_time):
        start_index = int(start_time * self.sample_rate) if start_time else 0
        end_index = int(end_time * self.sample_rate) if end_time else None
        self.cutoff_index(start_index, end_index)

    def low_pass_filter(self, signal, cutoff=5, order=2):
        return low_pass_filter(signal, self.sample_rate, cutoff, order)

    def plot_in_figure_pyplot(self, data, title, peaks=None):
        time = [i / self.sample_rate for i in range(len(data))]
        plt.figure()
        plt.plot(time, data)
        plt.title(title)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Value")
        if peaks is not None:
            peaks_x = [time[i] for i in peaks]
            peaks_y = [data[i] for i in peaks]
            plt.plot(peaks_x, peaks_y, "ro")

    def create_panda_df(self):
        return pd.DataFrame(
            {
                "Time (s)": np.arange(len(self.led_green_values)) / self.sample_rate,
                "LED Green": self.led_green_values,
                "LED IR": self.led_ir_values,
                "LED Red": self.led_red_values,
                "Ambient": self.ambient_values,
                "Accel X": self.acceleration_x,
                "Accel Y": self.acceleration_y,
                "Accel Z": self.acceleration_z,
                "Accel Euclidean": self.accel_euclidean,
                "Absolute Accel": self.absolute_accel,
                "Jerk Accel": self.jerk_accel / 60,
                "RMS Accel": self.rms_accel,
            }
        )
