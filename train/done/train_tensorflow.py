# train.py

import tensorflow as tf
from data.loader_tensor import load_split_from_csv, build_tf_dataset
from models.model_tensorflow import create_model
import argparse
import os

# -------------------------------
# Mixed Precision (AMP) Enablement
# -------------------------------
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("[INFO] Mixed Precision (AMP) enabled for GPU optimization.")

# -------------------------------
# Training Function
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Training pipeline for ML-JET multi-task classifier")
    parser.add_argument('--root_dir', type=str, required=True, help='Path to dataset root (containing splits)')
    parser.add_argument('--global_max', type=float, required=True, help='Global max for normalization')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Shuffle buffer size')
    parser.add_argument('--output_dir', type=str, default='checkpoints/', help='Directory to save models and logs')
    parser.add_argument('--model_tag', type=str, required=True, help='Unique model tag (e.g., EfficientNet, Transformer)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (optional, if configurable later)')

    args = parser.parse_args()

    # -------------------------------
    # Load Train/Val/Test Splits
    # -------------------------------
    print("[INFO] Loading dataset splits...")
    train_list = load_split_from_csv(os.path.join(args.root_dir, "train_files.csv"), args.root_dir)
    val_list = load_split_from_csv(os.path.join(args.root_dir, "val_files.csv"), args.root_dir)
    test_list = load_split_from_csv(os.path.join(args.root_dir, "test_files.csv"), args.root_dir)

    # -------------------------------
    # TensorFlow Dataset Pipeline
    # -------------------------------
    print("[INFO] Building TensorFlow datasets...")
    train_dataset = build_tf_dataset(train_list, args.global_max, batch_size=args.batch_size, buffer_size=args.buffer_size, shuffle=True)
    val_dataset = build_tf_dataset(val_list, args.global_max, batch_size=args.batch_size, buffer_size=args.buffer_size, shuffle=False)
    test_dataset = build_tf_dataset(test_list, args.global_max, batch_size=args.batch_size, buffer_size=args.buffer_size, shuffle=False)

    # -------------------------------
    # Load Model
    # -------------------------------
    print("[INFO] Building model...")
    model = create_model(learning_rate=args.learning_rate)
    model.build(input_shape=(None, 32, 32, 1))
    model.summary()

    # -------------------------------
    # Callbacks
    # -------------------------------
    # Dynamic output directory based on model and params
    run_tag = f"{args.model_tag}_bs{args.batch_size}_ep{args.epochs}_lr{args.learning_rate:.0e}"
    output_dir = os.path.join(args.output_dir, run_tag)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory for this run: {output_dir}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs')
        )
    ]

    # -------------------------------
    # Training Loop
    # -------------------------------
    print("[INFO] Starting training...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # After training, save summary
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"Model Tag: {args.model_tag}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Dataset Root: {args.root_dir}\n")
    # -------------------------------
    # Evaluation on Test Dataset
    # -------------------------------
    print("[INFO] Evaluating on test dataset...")
    results = model.evaluate(test_dataset)
    print("[RESULT] Test set performance:", results)

    print("✅ Training pipeline completed successfully.")
    # Optionally, print predictions for inspection
    # for batch_images, batch_labels in test_dataset.take(1):
    #     preds = model.predict(batch_images)
    #     print("Sample predictions (Energy Loss, αₛ, Q₀):", preds)
    #     print("True labels:", batch_labels)

# -------------------------------
# Entry Point for Command-Line
# -------------------------------
if __name__ == "__main__":
    main()

#Test Command
#python train.py --root_dir ~/hm_jetscapeml_source/data/jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled --global_max 121.79151153564453 --batch_size 512 --epochs 50 --output_dir training_output/

# python train.py \
# --root_dir ~/hm_jetscapeml_source/data/jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled \
# --global_max 121.79151153564453 \
# --batch_size 512 \
# --epochs 50 \
# --learning_rate 0.001 \
# --model_tag EfficientNet \
# --output_dir training_output/



