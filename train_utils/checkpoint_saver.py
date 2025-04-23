# train_utils/checkpoint_saver.py
import torch
import os

def save_epoch_checkpoint(epoch, model, optimizer, metrics, output_dir):
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, checkpoint_path)
    print(f"[INFO] Epoch {epoch+1}: 💾 Checkpoint saved: {checkpoint_path}")