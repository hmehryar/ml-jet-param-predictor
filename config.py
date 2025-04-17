# config.py
import os
import platform
import socket
import getpass


def get_config():
    # System/user based dataset path
    system = platform.system()
    release = platform.release().lower()
    hostname = socket.gethostname().lower()
    user = getpass.getuser().lower()

    if system == "Linux" and "wsl" in release and "arsalan" in user:
        base_path = "/mnt/d/Projects/110_JetscapeML/hm_jetscapeml_source/data"
    elif system == "Linux" and "ds044955" in hostname and "arsalan" in user:
        base_path = "/home/arsalan/Projects/110_JetscapeML/hm_jetscapeml_source/data"
    elif system == "Linux" and "gy4065" in user:
        base_path = "/wsu/home/gy/gy40/gy4065/hm_jetscapeml_source/data"
    else:
        raise RuntimeError("Unknown system configuration.")

    dataset_subdir = "jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled"
    dataset_root = os.path.join(base_path, dataset_subdir)

    return {
        "model_tag": "EfficientNet",
        "backbone": "efficientnet",
        "batch_size": 512,
        "epochs": 50,
        "learning_rate": 1e-3,
        "patience": 5,
        "input_shape": (1, 32, 32),
        "root_dir": dataset_root,
        "output_dir": os.path.join("checkpoints", "EfficientNet_bs512_ep50_lr1e-3_ds1000"),
        "train_csv": os.path.join(dataset_root, "train_files.csv"),
        "val_csv": os.path.join(dataset_root, "val_files.csv"),
    }
