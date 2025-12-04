import numpy as np

def euclidean_accel(signal_x, signal_y, signal_z):
    return np.sqrt(
        signal_x ** 2 + signal_y ** 2 + signal_z ** 2
    )

def sum_absolute_acceleration(signal_x, signal_y, signal_z):
    return np.abs(signal_x) + np.abs(signal_y) + np.abs(signal_z)


def jerk_based_acceleration(signal_x, signal_y, signal_z, sample_rate = 25):
    dax = np.diff(signal_x) * sample_rate
    day = np.diff(signal_y) * sample_rate
    daz = np.diff(signal_z) * sample_rate
    jerk = np.sqrt(dax ** 2 + day ** 2 + daz ** 2)
    return np.concatenate(([0], jerk))