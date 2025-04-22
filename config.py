# config.py
import os
import platform
import socket
import getpass
import re
from dataclasses import dataclass
import os

from types import SimpleNamespace


def get_config():
    # --- System/User Info ---
    system = platform.system()
    release = platform.release().lower()
    hostname = socket.gethostname().lower()
    user = getpass.getuser().lower()

    # --- Dataset Path Based on System ---
    if system == "Linux" and "wsl" in release and "arsi" in user:
        base_path = "/mnt/d/Projects/110_JetscapeML/hm_jetscapeml_source/data"
        print("[INFO] Detected WSL environment")
    elif system == "Linux" and "ds044955" in hostname and "arsalan" in user:
        base_path = "/home/arsalan/Projects/110_JetscapeML/hm_jetscapeml_source/data"
        print("[INFO] Detected native Ubuntu host: DS044955")
    elif system == "Linux" and "gy4065" in user:
        base_path = "/wsu/home/gy/gy40/gy4065/hm_jetscapeml_source/data"
        print("[INFO] Detected WSU Grid environment (user: gy4065)")
    else:
        raise RuntimeError("\u274c Unknown system. Please define the dataset path for this host.")

    # --- Dataset Subdir ---
    dataset_subdir = "jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled"
    dataset_root_dir = os.path.join(base_path, dataset_subdir)
    print(f"[INFO] Using dataset root: {dataset_root_dir}")

    # --- Extract dataset size from path ---
    match = re.search(r"size_(\d+)", dataset_root_dir)
    dataset_size = match.group(1) if match else "unknown"
    print(f"[INFO] Detected dataset size: {dataset_size}")

    # === Define all backbone-model_tag pairs to evaluate ===
    experiments = [
        {"model_tag": "EfficientNet", "backbone": "efficientnet"},
        {"model_tag": "ConvNeXt", "backbone": "convnext"},
        {"model_tag": "SwinTransformerV2", "backbone": "swin"},
        {"model_tag": "Mamba", "backbone": "mamba"},
        {"model_tag": "VisionMamba", "backbone": "vision_mamba"},
    ]
    model_idx = 0
    model_tag = experiments[model_idx]["model_tag"]
    backbone = experiments[model_idx]["backbone"]
    # ==========================================================
    print(f"[INFO] model tag: {model_tag}, backbone: {backbone}")

    input_shape=(1, 32, 32)
    batch_size = 512
    epochs = 50
    learning_rate = 1e-4
    patience= 5
    global_max = 121.79151153564453
    output_dir = 'training_output/'
    # Build dynamic output directory
    run_tag = f"{model_tag}_bs{batch_size}_ep{epochs}_lr{learning_rate:.0e}_ds{dataset_size}"
    

    # --- Config Dictionary ---
    return SimpleNamespace(**{
        "model_tag": model_tag,
        "backbone": backbone,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "patience": patience,
        "input_shape": input_shape,
        "global_max": global_max,
        "dataset_root_dir": dataset_root_dir,
        "train_csv": os.path.join(dataset_root_dir, "train_files.csv"),
        "val_csv": os.path.join(dataset_root_dir, "val_files.csv"),
        "test_csv": os.path.join(dataset_root_dir, "test_files.csv"),
        "output_dir": os.path.join(output_dir, run_tag)
    })
