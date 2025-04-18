# train_utils/resume.py
import datetime
import os
import torch
import json

def load_resume_state(output_dir, model, optimizer, device,config):
    summary_path = os.path.join(output_dir, "training_summary.json")
    resume_path = os.path.join(output_dir, "best_model.pth")
    best_model_path = os.path.join(output_dir, f"best_model.pth")

    patience = 5
    best_epoch = 0
    start_epoch = 0
    best_total_acc = 0.0
    early_stop_counter = 0

    best_metrics = {}
    training_summary= {}

    # Try loading resume state
    if os.path.exists(resume_path) and os.path.exists(summary_path):
        print(f"🔁 Resuming training from checkpoint and summary")
        
        # Load model checkpoint
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_total_acc = checkpoint.get('acc_total', 0.0)
        best_metrics = checkpoint.get('metrics', {})

        # Load summary info (optional counters/history)
        with open(summary_path, "r") as f:
            summary_data = json.load(f)
            early_stop_counter = summary_data.get("early_stop_counter", 0)
            acc_energy_list = summary_data.get("acc_energy_list", [])
            acc_alpha_list = summary_data.get("acc_alpha_list", [])
            acc_q0_list = summary_data.get("acc_q0_list", [])
            acc_total_list = summary_data.get("acc_total_list", [])
            all_epoch_metrics = summary_data.get("all_epoch_metrics", [])

        print(f"[INFO] Resumed at epoch {start_epoch} with total acc {best_total_acc:.4f} and early stop counter {early_stop_counter}")
    else:
        print(f"[INFO] Starting fresh training run")
        summary_path = initialize_training_summary(config, summary_path)
        print(f"[INFO] Initial training summary saved at: {summary_path}")
        
        
        
def initialize_training_summary(config, summary_path):
        # Initial summary structure
        training_summary = {
            "model_tag": config["model_tag"],
            "backbone": config["backbone"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "dataset_root": config["dataset_root_dir"],
            "global_max": config["global_max"],
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "interrupted_or_incomplete"
        }

        # Save the early config snapshot
        with open(summary_path, "w") as f:
            json.dump(training_summary, f, indent=2)
        return summary_path