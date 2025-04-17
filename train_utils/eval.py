# evaluate.py

import tensorflow as tf
from data.loader_tensor import load_split_from_csv, build_tf_dataset
from models.model_tensorflow import create_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import os
import json


# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_model(model_path, root_dir, global_max, batch_size=512, backbone='efficientnet', output_dir='evaluation_output'):
    # -------------------------------
    # 1. Load Test Dataset
    # -------------------------------
    print("[INFO] Loading test split...")
    test_list = load_split_from_csv(os.path.join(root_dir, "test_files.csv"), root_dir)
    test_dataset = build_tf_dataset(test_list, global_max, batch_size=batch_size, shuffle=False)

    # -------------------------------
    # 2. Load Model
    # -------------------------------
    print(f"[INFO] Loading model from checkpoint: {model_path}")
    model = create_model(backbone=backbone)
    model.load_weights(model_path)
    print("[INFO] Model loaded successfully.")

    # -------------------------------
    # 3. Predict on Test Set
    # -------------------------------
    print("[INFO] Running predictions on test set...")
    y_true = {'energy_loss_output': [], 'alpha_output': [], 'q0_output': []}
    y_pred = {'energy_loss_output': [], 'alpha_output': [], 'q0_output': []}

    for x_batch, y_batch in test_dataset:
        preds = model.predict(x_batch, verbose=0)

        for key in y_true.keys():
            # Append ground truth and predictions
            y_true[key].extend(np.argmax(y_batch[key], axis=1) if y_batch[key].shape[1] > 1 else y_batch[key].numpy().flatten())
            y_pred[key].extend(np.argmax(preds[key], axis=1) if preds[key].shape[1] > 1 else (preds[key] > 0.5).astype(int).flatten())

    # Convert lists to numpy arrays
    for key in y_true.keys():
        y_true[key] = np.array(y_true[key])
        y_pred[key] = np.array(y_pred[key])

    # -------------------------------
    # 4. Compute Metrics
    # -------------------------------
    print("[INFO] Computing evaluation metrics...")

    results = {}

    for key in y_true.keys():
        acc = accuracy_score(y_true[key], y_pred[key])
        report = classification_report(y_true[key], y_pred[key], output_dict=True)
        cm = confusion_matrix(y_true[key], y_pred[key])

        results[key] = {
            'accuracy': acc,
            'classification_report': report,
            'confusion_matrix': cm.tolist()  # Convert numpy array to list for JSON compatibility
        }

        # Print summary
        print(f"\n[RESULT] {key}:")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_true[key], y_pred[key]))
        print("Confusion Matrix:")
        print(cm)

    # -------------------------------
    # 5. Save Results
    # -------------------------------
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"evaluation_results_{backbone}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Evaluation results saved to {output_file}")


# -------------------------------
# Main CLI Function
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ML-JET model on test set")
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.h5 checkpoint')
    parser.add_argument('--root_dir', type=str, required=True, help='Dataset root directory (containing test_files.csv)')
    parser.add_argument('--global_max', type=float, required=True, help='Global max for normalization')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for evaluation (default: 512)')
    parser.add_argument('--backbone', type=str, default='efficientnet', help='Model backbone used (default: efficientnet)')
    parser.add_argument('--output_dir', type=str, default='evaluation_output', help='Directory to save evaluation results')

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        root_dir=args.root_dir,
        global_max=args.global_max,
        batch_size=args.batch_size,
        backbone=args.backbone,
        output_dir=args.output_dir
    )


# -------------------------------
# Example Usage
# -------------------------------
# python evaluate.py \
# --model_path training_output/EfficientNet_bs512_ep50_lr1e-03/best_model.h5 \
# --root_dir ~/hm_jetscapeml_source/data/jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled \
# --global_max 121.79151153564453 \
# --batch_size 512 \
# --backbone efficientnet \
# --output_dir evaluation_output/