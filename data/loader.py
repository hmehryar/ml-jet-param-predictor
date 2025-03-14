# loader.py

import os
import numpy as np
import tensorflow as tf

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

def tf_dataset_generator(file_label_list, global_min, global_max):
    """TensorFlow-compatible generator that yields normalized event data and labels."""
    for file_path, label in file_label_list:
        event = np.load(file_path).astype(np.float32)
        event = (event - global_min) / (global_max - global_min)  # Normalize
        event = np.expand_dims(event, axis=-1)  # Add channel dimension (32, 32, 1)
        yield event, np.array(label, dtype=np.int32)


# --------------------------------------------
# 4. TensorFlow Dataset Pipeline Construction
# --------------------------------------------

def build_tf_dataset(file_label_list, global_min, global_max, batch_size=512, buffer_size=10000, shuffle=True):
    """
    Build TensorFlow Dataset pipeline with shuffling, batching, and prefetching.

    Args:
        file_label_list (list): List of (file_path, label_tuple).
        global_min (float): Global min value for normalization.
        global_max (float): Global max value for normalization.
        batch_size (int): Batch size.
        buffer_size (int): Shuffle buffer size.
        shuffle (bool): Whether to shuffle dataset.

    Returns:
        tf.data.Dataset: Prepared dataset for training or evaluation.
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: tf_dataset_generator(file_label_list, global_min, global_max),
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
# 5. Optional: Helper to Split Files into Train/Val/Test
# ----------------------------------------------------------

def split_file_list(file_label_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split file list into train, validation, and test sets.

    Args:
        file_label_list (list): List of (file_path, label_tuple).
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.

    Returns:
        tuple: (train_list, val_list, test_list)
    """
    np.random.shuffle(file_label_list)
    total = len(file_label_list)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    return file_label_list[:train_end], file_label_list[train_end:val_end], file_label_list[val_end:]


# --------------------------
# Example Usage (Optional)
# --------------------------

if __name__ == "__main__":
    root_dir = "ML-JET-DATA/"  # Adjust to your dataset root
    global_min, global_max = 0, 500  # Example values — replace with actual min/max
    batch_size = 512

    # Generate file-label list
    file_label_list = list(file_label_generator(root_dir))

    # Split datasets
    train_list, val_list, test_list = split_file_list(file_label_list)

    # Build TensorFlow Datasets
    train_dataset = build_tf_dataset(train_list, global_min, global_max, batch_size=batch_size, shuffle=True)
    val_dataset = build_tf_dataset(val_list, global_min, global_max, batch_size=batch_size, shuffle=False)
    test_dataset = build_tf_dataset(test_list, global_min, global_max, batch_size=batch_size, shuffle=False)

    # Example: Check one batch
    for batch_images, batch_labels in train_dataset.take(1):
        print("Batch images shape:", batch_images.shape)
        print("Batch labels shape:", batch_labels.shape)
