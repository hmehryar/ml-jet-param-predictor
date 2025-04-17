# train.py

import os
from config import get_config
from models.model_factory import create_model
from data.loader import get_dataloaders
from train_utils.train_epoch import train_one_epoch
from train_utils.evaluate import evaluate
from train_utils.plot_metrics import plot_train_val_metrics
import torch
import json
import pandas as pd
import datetime


def main():
    cfg = get_config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Set seed, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = get_dataloaders(cfg)

    # Model and optimizer
    model, optimizer = create_model(cfg.backbone, cfg.input_shape, cfg.learning_rate)
    model.to(device)

    criterion = {
        'energy_loss_output': torch.nn.BCELoss(),
        'alpha_output': torch.nn.CrossEntropyLoss(),
        'q0_output': torch.nn.CrossEntropyLoss()
    }

    # Resume state
    start_epoch, best_total_acc, early_stop_counter, best_epoch, best_metrics, all_epoch_metrics = 0, 0.0, 0, 0, {}, []

    # Training trackers
    train_losses, val_losses = [], []
    train_acc_total_list, val_acc_total_list = [], []

    for epoch in range(start_epoch, cfg.epochs):
        train_metrics = train_one_epoch(train_loader, model, criterion, optimizer, device)
        val_metrics = evaluate(val_loader, model, criterion, device)

        # Log
        train_losses.append(train_metrics["train_loss"])
        val_losses.append(val_metrics["val_loss"])
        train_acc_total_list.append(train_metrics["train_acc_total"])
        val_acc_total_list.append(val_metrics["total_accuracy"])

        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{cfg.epochs} | "
              f"Train Loss: {train_metrics['train_loss']:.4f} | Val Acc: {val_metrics['total_accuracy']:.4f}")

        # Early stopping + best model save
        if val_metrics["total_accuracy"] > best_total_acc:
            best_total_acc = val_metrics["total_accuracy"]
            best_metrics = val_metrics
            best_epoch = epoch + 1
            early_stop_counter = 0

            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': best_metrics
            }, os.path.join(cfg.output_dir, "best_model.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
                break

        # Save epoch record
        epoch_record = {"epoch": epoch+1, **train_metrics, **val_metrics}
        all_epoch_metrics.append(epoch_record)

        with open(os.path.join(cfg.output_dir, "epoch_metrics.json"), "w") as f:
            json.dump(all_epoch_metrics, f, indent=2)

        pd.DataFrame(all_epoch_metrics).to_csv(os.path.join(cfg.output_dir, "epoch_metrics.csv"), index=False)

    # Plot final metrics
    plot_train_val_metrics(train_losses, val_losses, train_acc_total_list, val_acc_total_list, cfg.output_dir)


if __name__ == "__main__":
    main()
