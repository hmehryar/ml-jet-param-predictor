# train_utils/plot_metrics.py
import matplotlib.pyplot as plt
import os

def plot_train_val_metrics(train_losses, val_losses, train_accs, val_accs, output_dir):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot losses (solid)
    ax1.plot(epochs, train_losses, label="$Loss_{Train}$", color="black", linestyle='-')
    ax1.plot(epochs, val_losses, label="$Loss_{Val}$", color="black", linestyle='--')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # Plot accuracies (dashed)
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accs, label="$Acc_{Train}$", color="green", linestyle='-')
    ax2.plot(epochs, val_accs, label="$Acc_{Val}$", color="orange", linestyle='--')
    ax2.set_ylabel("Accuracy")

    # Title, grid, legend
    ax1.set_title("Train/Validation Loss and Accuracy Over Epochs")
    ax1.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()

    # Save
    fig_path_png = os.path.join(output_dir, "loss_accuracy_plot.png")
    fig_path_pdf = os.path.join(output_dir, "loss_accuracy_plot.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    print(f"📊 Plots saved to:\n - {fig_path_png}\n - {fig_path_pdf}")
    plt.close()
