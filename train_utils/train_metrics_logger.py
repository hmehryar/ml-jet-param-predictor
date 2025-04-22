import os
import json
import pandas as pd
def update_train_logs(train_metrics,
                      train_losses,
                      train_loss_energy_list,
                      train_loss_alpha_list,
                      train_loss_q0_list,
                      train_energy_acc_list,
                      train_alpha_acc_list,
                      train_q0_acc_list,
                      train_total_acc_list):

    train_losses.append(train_metrics['train_loss'])
    train_loss_energy_list.append(train_metrics['train_loss_energy'])
    train_loss_alpha_list.append(train_metrics['train_loss_alpha'])
    train_loss_q0_list.append(train_metrics['train_loss_q0'])

    train_energy_acc_list.append(train_metrics['train_acc_energy'])
    train_alpha_acc_list.append(train_metrics['train_acc_alpha'])
    train_q0_acc_list.append(train_metrics['train_acc_q0'])
    train_total_acc_list.append(train_metrics['train_acc'])

    return (train_losses,
            train_loss_energy_list,
            train_loss_alpha_list,
            train_loss_q0_list,
            train_energy_acc_list,
            train_alpha_acc_list,
            train_q0_acc_list,
            train_total_acc_list)
    

def update_val_logs(val_metrics,
                    val_losses,
                    val_loss_energy_list,
                    val_loss_alpha_list,
                    val_loss_q0_list,
                    acc_energy_list,
                    acc_alpha_list,
                    acc_q0_list,
                    acc_total_list):

    acc_total = val_metrics["val_accuracy"]
    acc_total_list.append(acc_total)

    acc_energy_list.append(val_metrics['energy']['accuracy'])
    acc_alpha_list.append(val_metrics['alpha']['accuracy'])
    acc_q0_list.append(val_metrics['q0']['accuracy'])

    val_loss = val_metrics["val_loss"]
    val_loss_energy = val_metrics["val_loss_energy"]
    val_loss_alpha = val_metrics["val_loss_alpha"]
    val_loss_q0 = val_metrics["val_loss_q0"]

    val_losses.append(val_loss)
    val_loss_energy_list.append(val_loss_energy)
    val_loss_alpha_list.append(val_loss_alpha)
    val_loss_q0_list.append(val_loss_q0)

    return (val_losses,
            val_loss_energy_list,
            val_loss_alpha_list,
            val_loss_q0_list,
            acc_energy_list,
            acc_alpha_list,
            acc_q0_list,
            acc_total_list)

def record_and_save_epoch(epoch, train_metrics, val_metrics, all_epoch_metrics, output_dir):
    epoch_record = {
        "epoch": epoch + 1,

        # Train losses
        "train_loss": train_metrics["train_loss"],
        "train_loss_energy": train_metrics["train_loss_energy"],
        "train_loss_alpha": train_metrics["train_loss_alpha"],
        "train_loss_q0": train_metrics["train_loss_q0"],

        # Train accuracies
        "train_acc_energy": train_metrics["train_acc_energy"],
        "train_acc_alpha": train_metrics["train_acc_alpha"],
        "train_acc_q0": train_metrics["train_acc_q0"],
        "train_acc": train_metrics["train_acc"],

        # Validation loss
        "val_loss": val_metrics["val_loss"],
        "val_loss_energy": val_metrics["val_loss_energy"],
        "val_loss_alpha": val_metrics["val_loss_alpha"],
        "val_loss_q0": val_metrics["val_loss_q0"],

        # Validation metrics
        "val_total_accuracy": val_metrics["val_accuracy"],
        "val_energy": val_metrics["energy"],
        "val_alpha": val_metrics["alpha"],
        "val_q0": val_metrics["q0"],
    }
    all_epoch_metrics.append(epoch_record)

    # Save to JSON
    with open(os.path.join(output_dir, "epoch_metrics.json"), "w") as f:
        json.dump(all_epoch_metrics, f, indent=2)

    # Save to CSV
    df = pd.DataFrame([
        {
            'epoch': m['epoch'],
            'train_loss': m['train_loss'],
            'train_loss_energy': m['train_loss_energy'],
            'train_loss_alpha': m['train_loss_alpha'],
            'train_loss_q0': m['train_loss_q0'],
            'train_acc_energy': m['train_acc_energy'],
            'train_acc_alpha': m['train_acc_alpha'],
            'train_acc_q0': m['train_acc_q0'],
            'train_acc': m['train_acc'],
            'val_loss': m['val_loss'],
            'val_loss_energy': m['val_loss_energy'],
            'val_loss_alpha': m['val_loss_alpha'],
            'val_loss_q0': m['val_loss_q0'],
            'val_acc': m['val_accuracy'],
            'val_energy_acc': m['energy']['accuracy'],
            'val_energy_prec': m['energy']['precision'],
            'val_energy_rec': m['energy']['recall'],
            'val_energy_f1': m['energy']['f1'],
            'val_alpha_acc': m['alpha']['accuracy'],
            'val_alpha_prec': m['alpha']['precision'],
            'val_alpha_rec': m['alpha']['recall'],
            'val_alpha_f1': m['alpha']['f1'],
            'val_q0_acc': m['q0']['accuracy'],
            'val_q0_prec': m['q0']['precision'],
            'val_q0_rec': m['q0']['recall'],
            'val_q0_f1': m['q0']['f1'],
        }
        for m in all_epoch_metrics
    ])
    df.to_csv(os.path.join(output_dir, "epoch_metrics.csv"), index=False)

    return all_epoch_metrics
