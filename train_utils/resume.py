# train_utils/resume.py
import datetime
import os
import torch
import json
from train_utils.training_summary import init_training_summary

def init_resume_state(model, optimizer, device,config):
    print(f"[INFO] Init Resume/Training Parameters")
    early_stop_counter,start_epoch,best_acc,best_epoch,best_metrics,all_epoch_metrics= 0, 0, 0.0, 0, {}, []

    summary_path = os.path.join(config.output_dir, "training_summary.json")
    resume_path = os.path.join(config.output_dir, "best_model.pth")

    # best_epoch = 0
    # start_epoch = 0
    # best_acc = 0.0
    # early_stop_counter = 0

    best_metrics = {}
    training_summary= {}

    # Try loading resume state
    if os.path.exists(resume_path) and os.path.exists(summary_path):
        print(f"[INFO] 🔁 Resuming training from checkpoint and summary")
        
        # Load model checkpoint
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_epoch = checkpoint['epoch']
        best_acc = checkpoint['metrics']['accuracy']
        best_metrics = checkpoint.get('metrics', {})

        # Load summary info (optional counters/history)
        with open(summary_path, "r") as f:
            training_summary = json.load(f)
            early_stop_counter = training_summary.get("early_stop_counter", 0)
            metric_file_path= training_summary.get("metrics_file", "")
            with open(metric_file_path, "r") as m:
                all_epoch_metrics = json.load(m)

        print(f"[INFO] Resumed at epoch {start_epoch} with total acc {best_acc:.4f} and early stop counter {early_stop_counter}")
    else:
        print(f"[INFO] Starting fresh training run by initializing training summary")
        training_summary=init_training_summary(config)
    return model, optimizer, start_epoch, best_acc, early_stop_counter, best_epoch, best_metrics, training_summary, all_epoch_metrics
        