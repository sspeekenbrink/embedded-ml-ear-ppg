import os


def get_available_directories(root_data_path):
    """
    Scans the root_data_path for subdirectories that contain both 'ecg.csv' and 'ppg.txt'.
    Returns a sorted list of valid directory names.
    """
    directories = []
    if os.path.exists(root_data_path) and os.path.isdir(root_data_path):
        for item in os.listdir(root_data_path):
            item_path = os.path.join(root_data_path, item)
            if os.path.isdir(item_path):
                ecg_file = os.path.join(item_path, "ecg.csv")
                ppg_file = os.path.join(item_path, "ppg.txt")
                if os.path.exists(ecg_file) and os.path.exists(ppg_file):
                    directories.append(item)
    return sorted(directories)
