# train_utils/plot_metrics.py
import matplotlib.pyplot as plt
import os
import numpy as np

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
    plt.show()
    plt.close()




def plot_loss_accuracy(loss_list,
                        loss_energy_list,
                        loss_alpha_list,
                        loss_q0_list,
                        acc_list,
                        acc_energy_list,
                        acc_alpha_list,
                        acc_q0_list,
                        output_dir,
                        title=""):

    epochs = np.arange(1, len(loss_list) + 1)
    colors = {
        "total": "black",
        "energy": "red",
        "alpha": "blue",
        "q0": "green",
    }

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Losses
    ax1.plot(epochs, loss_list, label="$Loss_{Total}$", color=colors["total"], linestyle='-')
    ax1.plot(epochs, loss_energy_list, label="$Loss_{Energy}$", color=colors["energy"], linestyle='-')
    ax1.plot(epochs, loss_alpha_list, label="$Loss_{{\\alpha}_s}$", color=colors["alpha"], linestyle='-')
    ax1.plot(epochs, loss_q0_list, label="$Loss_{Q_0}$", color=colors["q0"], linestyle='-')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Plot Accuracies
    ax2 = ax1.twinx()
    ax2.plot(epochs, acc_list, label="$Acc_{Total}$", color=colors["total"], linestyle='--')
    ax2.plot(epochs, acc_energy_list, label="$Acc_{Energy}$", color=colors["energy"], linestyle='--')
    ax2.plot(epochs, acc_alpha_list, label="$Acc_{{\\alpha}_s}$", color=colors["alpha"], linestyle='--')
    ax2.plot(epochs, acc_q0_list, label="$Acc_{Q_0}$", color=colors["q0"], linestyle='--')
    ax2.set_ylabel("Accuracy")

    # Title and Legend
    ax1.set_title(f"{title}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()

    
    plot_file_name = title.lower().replace(" ", "_").replace(":", "").replace(",", "").replace("(", "").replace(")", "")

    # Save Plots
    png_path = os.path.join(output_dir, f"{plot_file_name}_plot.png")
    pdf_path = os.path.join(output_dir, f"{plot_file_name}_plot.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"📉 Loss plot saved as:\n  - {png_path}\n  - {pdf_path}")
    plt.show()
    plt.close()

