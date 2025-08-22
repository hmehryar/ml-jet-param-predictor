# train_utils/evaluate.py
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix


# def evaluate(loader, model, criterion, device):
#     model.eval()
#     y_true = {'energy': [], 'alpha': [], 'q0': []}
#     y_pred = {'energy': [], 'alpha': [], 'q0': []}
#     correct_all = 0
#     total = 0

#     val_loss_total = 0.0
#     val_loss_energy = 0.0
#     val_loss_alpha = 0.0
#     val_loss_q0 = 0.0

#     with torch.no_grad():
#         for x, labels in loader:
#             x = x.to(device)
#             for key in labels:
#                 labels[key] = labels[key].to(device)

#             outputs = model(x)

#             # Energy loss: binary thresholding
#             # pred_energy = (outputs['energy_loss_output'] > 0.5).long().squeeze()
#             energy_logits = outputs['energy_loss_output'].squeeze()
#             pred_energy = (torch.sigmoid(energy_logits) > 0.5).long()
            
#             pred_alpha = torch.argmax(outputs['alpha_output'], dim=1)
#             pred_q0 = torch.argmax(outputs['q0_output'], dim=1)

#             gt_energy = labels['energy_loss_output'].squeeze()
#             gt_alpha = labels['alpha_output'].squeeze()
#             gt_q0 = labels['q0_output'].squeeze()

#             y_true['energy'].extend(labels['energy_loss_output'].squeeze().cpu().numpy())
#             y_true['alpha'].extend(labels['alpha_output'].squeeze().cpu().numpy())
#             y_true['q0'].extend(labels['q0_output'].squeeze().cpu().numpy())

#             y_pred['energy'].extend(pred_energy.cpu().numpy())
#             y_pred['alpha'].extend(pred_alpha.cpu().numpy())
#             y_pred['q0'].extend(pred_q0.cpu().numpy())

#             # Total accuracy = all 3 correct
#             correct_batch = ((pred_energy == gt_energy) &
#                              (pred_alpha == gt_alpha) &
#                              (pred_q0 == gt_q0)).sum().item()
#             correct_all += correct_batch
#             total += x.size(0)

#             # Compute loss per task
#             energy_out = outputs['energy_loss_output'].squeeze()
#             alpha_out = outputs['alpha_output']
#             q0_out = outputs['q0_output']

#             loss_energy = criterion['energy_loss_output'](energy_out, gt_energy.float())
#             loss_alpha = criterion['alpha_output'](alpha_out, gt_alpha)
#             loss_q0 = criterion['q0_output'](q0_out, gt_q0)

#             val_loss_energy += loss_energy.item()
#             val_loss_alpha += loss_alpha.item()
#             val_loss_q0 += loss_q0.item()
#             val_loss_total += (loss_energy + loss_alpha + loss_q0).item()

            
#     # Compute individual accuracies
#     acc_total = correct_all / total
#     avg_loss_energy = val_loss_energy / len(loader)
#     avg_loss_alpha = val_loss_alpha / len(loader)
#     avg_loss_q0 = val_loss_q0 / len(loader)
#     avg_val_loss = val_loss_total / len(loader)

#     # cm_energy = confusion_matrix(y_true["energy"], y_pred["energy"])
#     # cm_alpha = confusion_matrix(y_true["alpha"], y_pred["alpha"])
#     # cm_q0 = confusion_matrix(y_true["q0"], y_pred["q0"])

#     # All metrics + losses in one dict
#     metrics = {
#         "loss": avg_val_loss,
#         "loss_energy": avg_loss_energy,
#         "loss_alpha": avg_loss_alpha,
#         "loss_q0": avg_loss_q0,
#         "accuracy": acc_total,
#         "energy": {
            
#             "accuracy": accuracy_score(y_true['energy'], y_pred['energy']),
#             "precision": precision_score(y_true['energy'], y_pred['energy'], average='macro',zero_division=0),
#             "recall": recall_score(y_true['energy'], y_pred['energy'], average='macro'),
#             "f1": f1_score(y_true['energy'], y_pred['energy'], average='macro'),
#             "confusion_matrix": confusion_matrix(y_true["energy"], y_pred["energy"]).tolist(),
#             # "confusion_matrix": cm_energy.tolist()
#         },
#         "alpha": {
#             "accuracy": accuracy_score(y_true['alpha'], y_pred['alpha']),
#             "precision": precision_score(y_true['alpha'], y_pred['alpha'], average='macro',zero_division=0),
#             "recall": recall_score(y_true['alpha'], y_pred['alpha'], average='macro'),
#             "f1": f1_score(y_true['alpha'], y_pred['alpha'], average='macro'),
#             "confusion_matrix": confusion_matrix(y_true["alpha"], y_pred["alpha"]).tolist(),
#             # "confusion_matrix": cm_alpha.tolist()
#         },
#         "q0": {
#             "accuracy": accuracy_score(y_true['q0'], y_pred['q0']),
#             "precision": precision_score(y_true['q0'], y_pred['q0'], average='macro',zero_division=0),
#             "recall": recall_score(y_true['q0'], y_pred['q0'], average='macro'),
#             "f1": f1_score(y_true['q0'], y_pred['q0'], average='macro'),
#             "confusion_matrix": confusion_matrix(y_true["q0"], y_pred["q0"]).tolist(),
#             # "confusion_matrix": cm_q0.tolist()
#         }
#     }

#     return metrics

def evaluate(loader, model, criterion, device,*, make_alpha_fig=False, alpha_fig_path=None, alpha_class_names=("0.2","0.3","0.4")):
    model.eval()
    y_true = {'energy': [], 'alpha': [], 'q0': []}
    y_pred = {'energy': [], 'alpha': [], 'q0': []}
    correct_all = 0
    total = 0

    val_loss_total = 0.0
    val_loss_energy = 0.0
    val_loss_alpha = 0.0
    val_loss_q0 = 0.0

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            for key in labels:
                labels[key] = labels[key].to(device)

            outputs = model(x)

            # Energy loss: binary thresholding
            # pred_energy = (outputs['energy_loss_output'] > 0.5).long().squeeze()
            energy_logits = outputs['energy_loss_output'].squeeze()
            pred_energy = (torch.sigmoid(energy_logits) > 0.5).long()
            
            pred_alpha = torch.argmax(outputs['alpha_output'], dim=1)
            pred_q0 = torch.argmax(outputs['q0_output'], dim=1)

            gt_energy = labels['energy_loss_output'].squeeze()
            gt_alpha = labels['alpha_output'].squeeze()
            gt_q0 = labels['q0_output'].squeeze()

            y_true['energy'].extend(labels['energy_loss_output'].squeeze().cpu().numpy())
            y_true['alpha'].extend(labels['alpha_output'].squeeze().cpu().numpy())
            y_true['q0'].extend(labels['q0_output'].squeeze().cpu().numpy())

            y_pred['energy'].extend(pred_energy.cpu().numpy())
            y_pred['alpha'].extend(pred_alpha.cpu().numpy())
            y_pred['q0'].extend(pred_q0.cpu().numpy())

            # Total accuracy = all 3 correct
            correct_batch = ((pred_energy == gt_energy) &
                             (pred_alpha == gt_alpha) &
                             (pred_q0 == gt_q0)).sum().item()
            correct_all += correct_batch
            total += x.size(0)

            # Compute loss per task
            energy_out = outputs['energy_loss_output'].squeeze()
            alpha_out = outputs['alpha_output']
            q0_out = outputs['q0_output']

            loss_energy = criterion['energy_loss_output'](energy_out, gt_energy.float())
            loss_alpha = criterion['alpha_output'](alpha_out, gt_alpha)
            loss_q0 = criterion['q0_output'](q0_out, gt_q0)

            val_loss_energy += loss_energy.item()
            val_loss_alpha += loss_alpha.item()
            val_loss_q0 += loss_q0.item()
            val_loss_total += (loss_energy + loss_alpha + loss_q0).item()

            
    # Compute individual accuracies
    acc_total = correct_all / total
    avg_loss_energy = val_loss_energy / len(loader)
    avg_loss_alpha = val_loss_alpha / len(loader)
    avg_loss_q0 = val_loss_q0 / len(loader)
    avg_val_loss = val_loss_total / len(loader)

    # cm_energy = confusion_matrix(y_true["energy"], y_pred["energy"])
    # cm_alpha = confusion_matrix(y_true["alpha"], y_pred["alpha"])
    # cm_q0 = confusion_matrix(y_true["q0"], y_pred["q0"])

    # All metrics + losses in one dict
    metrics = {
        "loss": avg_val_loss,
        "loss_energy": avg_loss_energy,
        "loss_alpha": avg_loss_alpha,
        "loss_q0": avg_loss_q0,
        "accuracy": acc_total,
        "energy": {
            
            "accuracy": accuracy_score(y_true['energy'], y_pred['energy']),
            "precision": precision_score(y_true['energy'], y_pred['energy'], average='macro',zero_division=0),
            "recall": recall_score(y_true['energy'], y_pred['energy'], average='macro'),
            "f1": f1_score(y_true['energy'], y_pred['energy'], average='macro'),
            "confusion_matrix": confusion_matrix(y_true["energy"], y_pred["energy"]).tolist(),
            # "confusion_matrix": cm_energy.tolist()
        },
        "alpha": {
            "accuracy": accuracy_score(y_true['alpha'], y_pred['alpha']),
            "precision": precision_score(y_true['alpha'], y_pred['alpha'], average='macro',zero_division=0),
            "recall": recall_score(y_true['alpha'], y_pred['alpha'], average='macro'),
            "f1": f1_score(y_true['alpha'], y_pred['alpha'], average='macro'),
            "confusion_matrix": confusion_matrix(y_true["alpha"], y_pred["alpha"]).tolist(),
            # "confusion_matrix": cm_alpha.tolist()
        },
        "q0": {
            "accuracy": accuracy_score(y_true['q0'], y_pred['q0']),
            "precision": precision_score(y_true['q0'], y_pred['q0'], average='macro',zero_division=0),
            "recall": recall_score(y_true['q0'], y_pred['q0'], average='macro'),
            "f1": f1_score(y_true['q0'], y_pred['q0'], average='macro'),
            "confusion_matrix": confusion_matrix(y_true["q0"], y_pred["q0"]).tolist(),
            # "confusion_matrix": cm_q0.tolist()
        }
    }
    # --- NEW: optionally create α_s-focused figure here ---
    if make_alpha_fig:
        # If the caller gave a directory, make a default filename:
        if alpha_fig_path is not None and alpha_fig_path.endswith(os.sep):
            alpha_fig_path = os.path.join(alpha_fig_path, "alpha_hist_per_true_bin.png")

        plot_alpha_histograms(
            y_true_alpha=y_true['alpha'],
            y_pred_alpha=y_pred['alpha'],
            class_names=alpha_class_names,
            save_path=alpha_fig_path,
            show=False  # toggle True in notebooks if you want it displayed immediately
        )
        metrics["alpha_hist_path"] = alpha_fig_path  # report where it was saved (if any)
    

    # # --- NEW: optionally create α_s-focused figure here ---
    # if make_alpha_fig:
    #     # If the caller gave a directory, make a default filename:
    #     if alpha_fig_path is not None and alpha_fig_path.endswith(os.sep):
    #         alpha_fig_path = os.path.join(alpha_fig_path, "alpha_hist_per_true_bin.png")

    #     plot_alpha_histograms_from_arrays(
    #         y_true_alpha=y_true['alpha'],
    #         y_pred_alpha=y_pred['alpha'],
    #         class_names=alpha_class_names,
    #         save_path=alpha_fig_path,
    #         show=False  # toggle True in notebooks if you want it displayed immediately
    #     )
    #     metrics["alpha_hist_path"] = alpha_fig_path  # report where it was saved (if any)

    return metrics

import os
import numpy as np
import matplotlib.pyplot as plt
# --- NEW: α_s histogram plotter ---
def plot_alpha_histograms_from_arrays(
    y_true_alpha,
    y_pred_alpha,
    class_names=("0.2", "0.3", "0.4"),
    figsize=(12, 3.8),
    suptitle="Predicted αₛ distribution per true αₛ bin",
    save_path=None,
    show=False,
):
    """
    Make 3 bar charts (one per true αₛ class). Each chart shows how the model's
    αₛ predictions are distributed for that true bin.

    Args:
        y_true_alpha (array-like, shape [N]): ground-truth αₛ class indices {0,1,2}
        y_pred_alpha (array-like, shape [N]): predicted αₛ class indices {0,1,2}
        class_names (tuple[str]): display names for classes in index order
        figsize (tuple): figure size
        suptitle (str): figure title
        save_path (str|None): if set, saves PNG to this path (creates dirs)
        show (bool): if True, plt.show() (useful in notebooks)
    Returns:
        fig (matplotlib.figure.Figure): the created figure (also saved if save_path)
    """
    y_true_alpha = np.asarray(y_true_alpha)
    y_pred_alpha = np.asarray(y_pred_alpha)
    num_classes = len(class_names)

    # counts[true_class, pred_class] = number of samples
    counts = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_alpha, y_pred_alpha):
        counts[int(t), int(p)] += 1

    fig, axes = plt.subplots(1, num_classes, figsize=figsize, sharey=True)
    if num_classes == 1:
        axes = [axes]

    x = np.arange(num_classes)
    for t in range(num_classes):
        ax = axes[t]
        ax.bar(x, counts[t], edgecolor="black")
        ax.set_xticks(x, class_names, rotation=0)
        ax.set_xlabel("Predicted αₛ")
        if t == 0:
            ax.set_ylabel("# of predictions")
        ax.set_title(f"True αₛ = {class_names[t]}\n(N={counts[t].sum()})")
        # annotate counts on top of bars
        for xi, c in zip(x, counts[t]):
            ax.text(xi, c, str(int(c)), ha="center", va="bottom", fontsize=9)

        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(suptitle, y=1.05, fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig

import matplotlib.pyplot as plt
import numpy as np

# def plot_alpha_histograms(y_true_alpha, y_pred_alpha, class_names=[r"$\alpha_s=0.2$", r"$\alpha_s=0.3$", r"$\alpha_s=0.4$"]):
#     """
#     Plot normalized histograms of predicted alpha_s values for each true alpha_s class.
#     Each histogram is normalized (sum=1), but also shows raw bin counts as annotations.
#     """

#     y_true_alpha = np.array(y_true_alpha)
#     y_pred_alpha = np.array(y_pred_alpha)
#     num_classes = len(class_names)

#     fig, axes = plt.subplots(1, num_classes, figsize=(15, 4), sharey=True)

#     for i, ax in enumerate(axes):
#         # Select predictions where ground truth == i
#         mask = (y_true_alpha == i)
#         preds = y_pred_alpha[mask]

#         # Count raw occurrences
#         counts, bins = np.histogram(preds, bins=np.arange(num_classes+1)-0.5)
#         total = counts.sum()

#         # Normalize counts (avoid div/0)
#         normalized = counts / total if total > 0 else counts

#         # Plot normalized histogram
#         ax.bar(range(num_classes), normalized, tick_label=class_names, alpha=0.7, color='C0', edgecolor='black')

#         # Add annotations with raw counts
#         for j, c in enumerate(counts):
#             ax.text(j, normalized[j] + 0.01, str(c), ha='center', va='bottom', fontsize=9)

#         ax.set_title(f"True {class_names[i]} (N={total})")
#         ax.set_ylabel("Normalized Frequency")

#     plt.tight_layout()
#     plt.show()
import matplotlib.pyplot as plt
import numpy as np

def plot_alpha_histograms(
                        y_true_alpha,
                        y_pred_alpha,
                        class_names=[r"$\alpha_s=0.2$", r"$\alpha_s=0.3$", r"$\alpha_s=0.4$"],
                        figsize=(17, 5),
                        suptitle="Predicted αₛ distribution per true αₛ bin",
                        save_path=None,
                        show=False,
                      ):
    """
    Plot normalized histograms of predicted alpha_s values for each true alpha_s class.
    Bars are normalized (sum=1) but annotated with raw count and probability.
    """

    y_true_alpha = np.array(y_true_alpha)
    y_pred_alpha = np.array(y_pred_alpha)
    num_classes = len(class_names)

    fig, axes = plt.subplots(1, num_classes, figsize=figsize, sharey=True)

    for i, ax in enumerate(axes):
        # Select predictions where ground truth == i
        mask = (y_true_alpha == i)
        preds = y_pred_alpha[mask]

        # Count raw occurrences
        counts = np.bincount(preds, minlength=num_classes)
        total = counts.sum()

        # Normalize counts to probabilities
        probs = counts / total if total > 0 else np.zeros_like(counts)

        # Plot normalized histogram
        ax.bar(range(num_classes), probs, tick_label=class_names,
               alpha=0.7, color='C0', edgecolor='black')

        # Add annotations with raw counts + probability
        for j, (c, p) in enumerate(zip(counts, probs)):
            ax.text(j, p + 0.01, f"{c} ({p:.2f})",
                    ha='center', va='bottom', fontsize=9)

        ax.set_title(f"True {class_names[i]} (N={total})")
        ax.set_ylabel("Normalized Frequency")

    fig.suptitle(suptitle, y=1.05, fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.tight_layout()
        plt.show()

    return fig
