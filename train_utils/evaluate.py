# train_utils/evaluate.py
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import os

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



def evaluate(loader, model, criterion, device,*,
              make_alpha_fig=False, alpha_fig_path=None, alpha_class_names=("0.2","0.3","0.4"),
              make_alpha_avgprob_fig=False, alpha_avgprob_fig_path=None):
    model.eval()
    y_true = {'energy': [], 'alpha': [], 'q0': []}
    y_pred = {'energy': [], 'alpha': [], 'q0': []}
    
    # NEW: collect α_s probabilities so we can pass to the prob-hist method
    alpha_proba_rows = []   # will become [N, C] numpy

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

            alpha_logits = outputs['alpha_output']
            pred_alpha = torch.argmax(alpha_logits, dim=1)
            alpha_proba = torch.softmax(alpha_logits, dim=1)
            alpha_proba_rows.append(alpha_proba.cpu().numpy())

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

    # Aggregate α_s probabilities (N, C)
    if len(alpha_proba_rows):
        y_alpha_proba = np.concatenate(alpha_proba_rows, axis=0)
    else:
        y_alpha_proba = np.zeros((0, len(alpha_class_names)), dtype=float)

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
        },
        # expose α_s soft info so notebooks can reuse without re-running evaluation
        "alpha_soft": {
            "y_true_alpha": np.asarray(y_true['alpha']).tolist(),
            "y_alpha_proba": y_alpha_proba.tolist(),
            "class_names": list(alpha_class_names),
        }
    }
    # --- NEW: optionally create α_s-focused figure here ---
    if make_alpha_fig:
        
        plot_alpha_histograms(
            y_true_alpha=y_true['alpha'],
            y_pred_alpha=y_pred['alpha'],
            class_names=[rf"$\alpha_s={c}$" for c in alpha_class_names],
            save_path=alpha_fig_path,
            show=False  # toggle True in notebooks if you want it displayed immediately
        )

    if make_alpha_avgprob_fig:
        metrics["alpha_hist_path"] = alpha_fig_path  # report where it was saved (if any)
        plot_alpha_avgprob_histograms(
            y_true_alpha=y_true['alpha'],
            y_alpha_proba=y_alpha_proba,
            class_names=[rf"$\alpha_s={c}$" for c in alpha_class_names],
            save_path=alpha_avgprob_fig_path,
            show=False
        )
        metrics["alpha_avgprob_hist_path"] = (alpha_avgprob_fig_path + ".png") if alpha_avgprob_fig_path else None

    return metrics

# ---------------------------
# α_s helpers (probabilities)
# ---------------------------

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

    y_true_alpha = np.array(y_true_alpha)
    y_pred_alpha = np.array(y_pred_alpha)
    num_classes = len(class_names)

    fig, axes = plt.subplots(1, num_classes, figsize=figsize, sharey=True,constrained_layout=True)

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
            ax.text(j, p + 0.02, f"{c} ({p:.2f})",
                    ha='center', va='bottom', fontsize=10, clip_on=False)
        
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        ax.set_title(f"True {class_names[i]} (N={total})")
        ax.set_ylabel("Normalized Frequency")

    fig.suptitle(suptitle, y=1.05, fontsize=12)
    fig.tight_layout()


    if save_path is not None:
        base = os.path.splitext(save_path)[0]  # allow ".png" or path without ext
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")
    if show:
        plt.tight_layout()
        plt.show()

    return fig



def plot_alpha_avgprob_histograms(
    y_true_alpha,
    y_alpha_proba,                        # shape (N, C), softmax probs per sample
    class_names=(r"$\alpha_s=0.2$", r"$\alpha_s=0.3$", r"$\alpha_s=0.4$"),
    figsize=(17, 5.5),
    suptitle=r"Average predicted $\alpha_s$ probability per true $\alpha_s$",
    save_path=None,
    show=False,
):
    """
    For each TRUE α_s class t, compute the mean predicted probability vector over all its samples:
        avg_probs[t] = mean( y_alpha_proba[ y_true_alpha == t ] , axis=0 )
    Then plot 3 bar charts (one per true α_s), bars = average probs for predicted classes.
    """
    y_true_alpha = np.asarray(y_true_alpha, dtype=int)
    y_alpha_proba = np.asarray(y_alpha_proba, dtype=float)
    C = len(class_names)

    # compute avg probs per true class
    avg_probs = np.zeros((C, C), dtype=float)
    Ns = np.zeros(C, dtype=int)
    for t in range(C):
        mask = (y_true_alpha == t)
        Ns[t] = int(mask.sum())
        if Ns[t] > 0:
            avg_probs[t] = y_alpha_proba[mask].mean(axis=0)

    # figure
    fig, axes = plt.subplots(1, C, figsize=figsize, sharey=True, constrained_layout=True)
    if C == 1:
        axes = [axes]
    x = np.arange(C)

    # global headroom to avoid label clipping
    ymax = float(np.max(avg_probs)) if avg_probs.size else 1.0
    ylim_top = max(0.05, min(1.05, ymax + 0.12))

    for t, ax in enumerate(axes):
        bars = ax.bar(x, avg_probs[t], edgecolor="black", alpha=0.85, tick_label=class_names)
        ax.set_ylim(0, ylim_top)
        ax.set_title(f"True {class_names[t]} (N={Ns[t]})")
        if t == 0:
            ax.set_ylabel("Average probability")

        # annotate value on each bar
        for j, b in enumerate(bars):
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h + 0.02, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=10, clip_on=False)

        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(suptitle, y=1.05, fontsize=12)

    if save_path:
        base = os.path.splitext(save_path)[0]  # allow ".png" or path without ext
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")

    if show:
        plt.show()
    return fig
