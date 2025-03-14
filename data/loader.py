# loader.py

import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
import argparse

# -------------------------------
# 1. Directory to Label Mapping
# -------------------------------

alpha_map = {"0.2": 0, "0.3": 1, "0.4": 2}
q0_map = {"1": 0, "1.5": 1, "2.0": 2, "2.5": 3}


def parse_labels_from_dir(dir_name):
    """Parse directory name to label tuple (energy_loss, alpha_s, q0)."""
    energy_loss_str, alpha_str, q0_str = dir_name.split('_')
    energy_loss = 0 if energy_loss_str == "MMAT" else 1
    alpha = alpha_map[alpha_str]
    q0 = q0_map[q0_str]
    return (energy_loss, alpha, q0)


# -----------------------------------------
# 2. File and Label Generator (Python level)
# -----------------------------------------

def file_label_generator(root_dir):
    """Yield (file_path, label_tuple) from dataset directory structure."""
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            label_tuple = parse_labels_from_dir(dir_name)
            for file_name in os.listdir(dir_path):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(dir_path, file_name)
                    yield file_path, label_tuple


# ---------------------------------------------------
# 3. TensorFlow Dataset Generator (Lazy loading .npy)
# ---------------------------------------------------

def tf_dataset_generator(file_label_list, global_max):
    """TensorFlow-compatible generator that yields normalized event data and labels."""
    for file_path, label in file_label_list:
        event = np.load(file_path).astype(np.float32)
        event = event / global_max  # Normalize assuming global_min = 0
        event = np.expand_dims(event, axis=-1)  # Add channel dimension (32, 32, 1)
        yield event, np.array(label, dtype=np.int32)


# --------------------------------------------
# 4. TensorFlow Dataset Pipeline Construction
# --------------------------------------------

def build_tf_dataset(file_label_list, global_max, batch_size=512, buffer_size=10000, shuffle=True):
    """
    Build TensorFlow Dataset pipeline with shuffling, batching, and prefetching.
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: tf_dataset_generator(file_label_list, global_max),
        output_signature=(
            tf.TensorSpec(shape=(32, 32, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.int32)
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# ----------------------------------------------------------
# 5. Stratified Split for Train/Val/Test with Random Seed
# ----------------------------------------------------------

def split_file_list(file_label_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Perform a stratified split on (file_path, label_tuple) into train, validation, and test sets.
    """
    np.random.seed(random_seed)

    label_to_files = defaultdict(list)
    for file_path, label in file_label_list:
        label_to_files[label].append(file_path)

    train_list, val_list, test_list = [], [], []

    for label, files in label_to_files.items():
        files = np.array(files)
        np.random.shuffle(files)
        total = len(files)

        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)

        if total < 3:
            train_split = files[:1]
            val_split = files[1:2] if total > 1 else []
            test_split = files[2:] if total > 2 else []
        else:
            train_split = files[:train_end]
            val_split = files[train_end:val_end]
            test_split = files[val_end:]

        train_list.extend([(fp, label) for fp in train_split])
        val_list.extend([(fp, label) for fp in val_split])
        test_list.extend([(fp, label) for fp in test_split])

    np.random.shuffle(train_list)
    np.random.shuffle(val_list)
    np.random.shuffle(test_list)

    return train_list, val_list, test_list


# --------------------------
# 6. Main CLI Entry Point
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorFlow DataLoader for ML-JET dataset")
    parser.add_argument('--root_dir', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--global_max', type=float, required=True, help='Global max for normalization')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for DataLoader')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Buffer size for shuffling')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    root_dir = args.root_dir
    global_max = args.global_max
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    random_seed = args.random_seed

    file_label_list = list(file_label_generator(root_dir))
    print(f"Total files found: {len(file_label_list)}")
    if file_label_list:
        print("Example file-label pair:", file_label_list[0])

    # Perform stratified split
    train_list, val_list, test_list = split_file_list(file_label_list, random_seed=random_seed)

    # Build TensorFlow Datasets
    train_dataset = build_tf_dataset(train_list, global_max, batch_size=batch_size, buffer_size=buffer_size, shuffle=True)
    val_dataset = build_tf_dataset(val_list, global_max, batch_size=batch_size, buffer_size=buffer_size, shuffle=False)
    test_dataset = build_tf_dataset(test_list, global_max, batch_size=batch_size, buffer_size=buffer_size, shuffle=False)

    # Example: Check one batch
    for batch_images, batch_labels in train_dataset.take(1):
        print("Batch images shape:", batch_images.shape)
        print("Batch labels shape:", batch_labels.shape)
