# train_utils/train_epoch.py
import torch

def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    loss_energy_total = 0.0
    loss_alpha_total = 0.0
    loss_q0_total = 0.0

    correct_energy = 0
    correct_alpha = 0
    correct_q0 = 0
    correct_all = 0
    total = 0

    for x, labels in loader:
        x = x.to(device)
        for key in labels:
            labels[key] = labels[key].to(device)

        outputs = model(x)
        energy_out = outputs['energy_loss_output'].squeeze()
        alpha_out = outputs['alpha_output']
        q0_out = outputs['q0_output']

        gt_energy = labels['energy_loss_output'].squeeze()
        gt_alpha = labels['alpha_output'].squeeze()
        gt_q0 = labels['q0_output'].squeeze()

        loss_energy = criterion['energy_loss_output'](energy_out, gt_energy.float())
        loss_alpha = criterion['alpha_output'](alpha_out, gt_alpha)
        loss_q0 = criterion['q0_output'](q0_out, gt_q0)
        total_batch_loss = loss_energy + loss_alpha + loss_q0

        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()
        loss_energy_total += loss_energy.item()
        loss_alpha_total += loss_alpha.item()
        loss_q0_total += loss_q0.item()

        pred_energy = (energy_out > 0.5).long()
        pred_alpha = torch.argmax(alpha_out, dim=1)
        pred_q0 = torch.argmax(q0_out, dim=1)

        correct_energy += (pred_energy == gt_energy).sum().item()
        correct_alpha += (pred_alpha == gt_alpha).sum().item()
        correct_q0 += (pred_q0 == gt_q0).sum().item()
        correct_all += ((pred_energy == gt_energy) &
                        (pred_alpha == gt_alpha) &
                        (pred_q0 == gt_q0)).sum().item()
        total += x.size(0)

    return {
        'train_loss': total_loss / len(loader),
        'train_loss_energy': loss_energy_total / len(loader),
        'train_loss_alpha': loss_alpha_total / len(loader),
        'train_loss_q0': loss_q0_total / len(loader),
        'train_acc_energy': correct_energy / total,
        'train_acc_alpha': correct_alpha / total,
        'train_acc_q0': correct_q0 / total,
        'train_acc_total': correct_all / total,
    }
