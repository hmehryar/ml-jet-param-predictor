# config.py
import os
import platform
import socket
import getpass
import re
import argparse
import yaml
from types import SimpleNamespace

def get_config(config_path=None):
    # --- Parse command-line args only if not running in notebook ---
    if config_path is None:
        try:
        # --- Parse command-line args ---
            parser = argparse.ArgumentParser(description="ML-JET Model Config")
            parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
            args = parser.parse_args()
            config_path = args.config
        except SystemExit:
            raise ValueError("Must provide config_path explicitly when running in notebook mode.")

    print(f"[INFO] Config Path: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # --- System/User Info ---
    system = platform.system()
    release = platform.release().lower()
    hostname = socket.gethostname().lower()
    user = getpass.getuser().lower()

    # --- Dataset Path Based on System ---
    if system == "Linux" and "hm-srv1" in hostname and "arsalan" in user and "wsl" in release :
        base_path = "/mnt/d/Projects/110_JetscapeML/hm_jetscapeml_source/data"
        print("[INFO] Detected WSL environment")
    elif system == "Linux" and "ds044955" in hostname and "arsalan" in user:
        base_path = "/home/arsalan/Projects/110_JetscapeML/hm_jetscapeml_source/data"
        # base_path = "/home/arsalan/wsu-grid/hm_jetscapeml_source/data/"
        print("[INFO] Detected native Ubuntu host: DS044955")
    elif system == "Linux" and "gy4065" in user:
        base_path = "/wsu/home/gy/gy40/gy4065/hm_jetscapeml_source/data"
        print("[INFO] Detected WSU Grid environment (user: gy4065)")
    else:
        raise RuntimeError("\u274c Unknown system. Please define the dataset path for this host.")

    # --- Dataset Subdir ---
    dataset_subdir = cfg_dict.get("dataset_subdir")
    dataset_root_dir = os.path.join(base_path, dataset_subdir)
    print(f"[INFO] Using dataset root: {dataset_root_dir}")

    # --- Dataset Size ---
    if "dataset_size" in cfg_dict:
        dataset_size = int(cfg_dict["dataset_size"])
        print(f"[INFO] Using dataset_size from config: {dataset_size}")
    else:
        match = re.search(r"size_(\d+)", dataset_root_dir)
        dataset_size = match.group(1) if match else "unknown"
        print(f"[INFO] Extracted dataset_size from path: {dataset_size}")

    # match = re.search(r"size_(\d+)", dataset_root_dir)
    # dataset_size = match.group(1) if match else "unknown"
    # print(f"[INFO] Detected dataset size: {dataset_size}")

    # --- Model Configs ---
    model_tag = cfg_dict.get("model_tag")
    backbone = cfg_dict.get("backbone")
    input_shape = tuple(cfg_dict.get("input_shape", [1, 32, 32]))
    batch_size = cfg_dict.get("batch_size", 512)
    epochs = cfg_dict.get("epochs", 50)
    learning_rate = cfg_dict.get("learning_rate", 1e-4)
    patience = cfg_dict.get("patience", 5)
    global_max = cfg_dict.get("global_max", 121.79151153564453)
    output_base = cfg_dict.get("output_dir", "training_output/")
    group_size    = cfg_dict.get("group_size", 1)

    # Set default scheduler settings if not present
    scheduler_defaults = {
        'type': 'ReduceLROnPlateau',
        'mode': 'max',
        'factor': 0.5,
        'patience': 4,
        'verbose': True
    }
    scheduler=cfg_dict.get('scheduler', scheduler_defaults)
    preloaded = ""
    preload_model_path = cfg_dict.get("preload_model_path")
    if preload_model_path:
        preloaded="_preloaded"

    # after scheduler / preload_model_path parsing
    loss_cfg = cfg_dict.get("loss", {}) or {}
    loss_weights = (loss_cfg.get("weights") or {
        "energy_loss_output": 1.0,
        "alpha_output": 1.0,
        "q0_output": 1.0,
    })
    if not (loss_cfg.get("weights") is None):
        # concatenate loss weights into a string and separate them with __annotations__
        weighted_loss="_weighted_loss{}".format("_".join([f"{k}_{v}" for k, v in loss_weights.items()]))
    else:
        weighted_loss = ""
    # --- Split CSV Paths based on group_size ---
    if group_size > 1 :
        basename     = f"file_labels_aggregated_ds{dataset_size}_g{group_size}"
    else:
        basename     = "file_labels"
    train_csv    = os.path.join(dataset_root_dir, f"{basename}_train.csv")
    val_csv      = os.path.join(dataset_root_dir, f"{basename}_val.csv")
    test_csv     = os.path.join(dataset_root_dir, f"{basename}_test.csv")

    scheduler_type = scheduler.get('type', 'NoScheduler')
    run_tag = f"{model_tag}_bs{batch_size}_ep{epochs}_lr{learning_rate:.0e}_ds{dataset_size}_g{group_size}_sched_{scheduler_type}{preloaded}{weighted_loss}"
    output_dir = os.path.join(output_base, run_tag)

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
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
        "output_dir": output_dir,
        "group_size": group_size,
        "scheduler": scheduler,
        "dataset_size": dataset_size,
        "preload_model_path": preload_model_path,
        "loss_weights": loss_weights,

    })

if __name__ == "__main__":
    cfg=get_config()
    print(cfg)

# # config.py
# import os
# import platform
# import socket
# import getpass
# import re
# from dataclasses import dataclass
# import os

# from types import SimpleNamespace


# def get_config():
#     # --- System/User Info ---
#     system = platform.system()
#     release = platform.release().lower()
#     hostname = socket.gethostname().lower()
#     user = getpass.getuser().lower()

#     # --- Dataset Path Based on System ---
#     if system == "Linux" and "wsl" in release and "arsi" in user:
#         base_path = "/mnt/d/Projects/110_JetscapeML/hm_jetscapeml_source/data"
#         print("[INFO] Detected WSL environment")
#     elif system == "Linux" and "ds044955" in hostname and "arsalan" in user:
#         base_path = "/home/arsalan/Projects/110_JetscapeML/hm_jetscapeml_source/data"
#         print("[INFO] Detected native Ubuntu host: DS044955")
#     elif system == "Linux" and "gy4065" in user:
#         base_path = "/wsu/home/gy/gy40/gy4065/hm_jetscapeml_source/data"
#         print("[INFO] Detected WSU Grid environment (user: gy4065)")
#     else:
#         raise RuntimeError("\u274c Unknown system. Please define the dataset path for this host.")

#     # --- Dataset Subdir ---
#     dataset_subdir = "jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled"
#     dataset_root_dir = os.path.join(base_path, dataset_subdir)
#     print(f"[INFO] Using dataset root: {dataset_root_dir}")

#     # --- Extract dataset size from path ---
#     match = re.search(r"size_(\d+)", dataset_root_dir)
#     dataset_size = match.group(1) if match else "unknown"
#     print(f"[INFO] Detected dataset size: {dataset_size}")

#     # === Define all backbone-model_tag pairs to evaluate ===
#     experiments = [
#         {"model_tag": "EfficientNet", "backbone": "efficientnet"},
#         {"model_tag": "ConvNeXt", "backbone": "convnext"},
#         {"model_tag": "SwinTransformerV2", "backbone": "swin"},
#         {"model_tag": "Mamba", "backbone": "mamba"},
#         {"model_tag": "VisionMamba", "backbone": "vision_mamba"},
#     ]
#     model_idx = 0
#     model_tag = experiments[model_idx]["model_tag"]
#     backbone = experiments[model_idx]["backbone"]
#     # ==========================================================
#     print(f"[INFO] model tag: {model_tag}, backbone: {backbone}")

#     input_shape=(1, 32, 32)
#     batch_size = 512
#     epochs = 50
#     learning_rate = 1e-4
#     patience= 5
#     global_max = 121.79151153564453
#     output_dir = 'training_output/'
#     # Build dynamic output directory
#     run_tag = f"{model_tag}_bs{batch_size}_ep{epochs}_lr{learning_rate:.0e}_ds{dataset_size}"
    

#     # --- Config Dictionary ---
#     return SimpleNamespace(**{
#         "model_tag": model_tag,
#         "backbone": backbone,
#         "batch_size": batch_size,
#         "epochs": epochs,
#         "learning_rate": learning_rate,
#         "patience": patience,
#         "input_shape": input_shape,
#         "global_max": global_max,
#         "dataset_root_dir": dataset_root_dir,
#         "train_csv": os.path.join(dataset_root_dir, "train_files.csv"),
#         "val_csv": os.path.join(dataset_root_dir, "val_files.csv"),
#         "test_csv": os.path.join(dataset_root_dir, "test_files.csv"),
#         "output_dir": os.path.join(output_dir, run_tag)
#     })

