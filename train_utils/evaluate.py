# train_utils/evaluate.py
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(loader, model, criterion, device):
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
            pred_energy = (outputs['energy_loss_output'] > 0.5).long().squeeze()
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
        },
        "alpha": {
            "accuracy": accuracy_score(y_true['alpha'], y_pred['alpha']),
            "precision": precision_score(y_true['alpha'], y_pred['alpha'], average='macro',zero_division=0),
            "recall": recall_score(y_true['alpha'], y_pred['alpha'], average='macro'),
            "f1": f1_score(y_true['alpha'], y_pred['alpha'], average='macro'),
        },
        "q0": {
            "accuracy": accuracy_score(y_true['q0'], y_pred['q0']),
            "precision": precision_score(y_true['q0'], y_pred['q0'], average='macro',zero_division=0),
            "recall": recall_score(y_true['q0'], y_pred['q0'], average='macro'),
            "f1": f1_score(y_true['q0'], y_pred['q0'], average='macro'),
        }
    }

    return metrics
