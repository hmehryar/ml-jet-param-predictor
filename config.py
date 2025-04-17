# config.py
import os
import platform
import socket
import getpass
import re

def get_config():
    # --- System/User Info ---
    system = platform.system()
    release = platform.release().lower()
    hostname = socket.gethostname().lower()
    user = getpass.getuser().lower()

    # --- Dataset Path Based on System ---
    if system == "Linux" and "wsl" in release and "arsi" in user:
        base_path = "/mnt/d/Projects/110_JetscapeML/hm_jetscapeml_source/data"
    elif system == "Linux" and "ds044955" in hostname and "arsalan" in user:
        base_path = "/home/arsalan/Projects/110_JetscapeML/hm_jetscapeml_source/data"
    elif system == "Linux" and "gy4065" in user:
        base_path = "/wsu/home/gy/gy40/gy4065/hm_jetscapeml_source/data"
    else:
        raise RuntimeError("\u274c Unknown system. Please define the dataset path for this host.")

    # --- Dataset Subdir ---
    dataset_subdir = "jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled"
    dataset_root_dir = os.path.join(base_path, dataset_subdir)

    # --- Extract dataset size from path ---
    match = re.search(r"size_(\d+)", dataset_root_dir)
    dataset_size = match.group(1) if match else "unknown"

    # --- Config Dictionary ---
    return {
        "model_tag": "EfficientNet",
        "backbone": "efficientnet",
        "batch_size": 512,
        "epochs": 50,
        "learning_rate": 1e-3,
        "patience": 5,
        "input_shape": (1, 32, 32),
        "global_max": 121.79151153564453,
        "dataset_root_dir": dataset_root_dir,
        "train_csv": os.path.join(dataset_root_dir, "train_files.csv"),
        "val_csv": os.path.join(dataset_root_dir, "val_files.csv"),
        "test_csv": os.path.join(dataset_root_dir, "test_files.csv"),
        "output_dir": os.path.join("checkpoints", f"EfficientNet_bs{512}_ep{50}_lr{1e-3}_ds{dataset_size}")
    }
