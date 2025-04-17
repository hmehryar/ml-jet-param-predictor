# model.py

import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------------------------
# Multi-Head Classifier Model with Flexible Backbone
# ---------------------------

class MultiHeadClassifier(Model):
    def __init__(self, backbone='efficientnet', input_shape=(32, 32, 1)):
        super(MultiHeadClassifier, self).__init__()

        # ----------------------
        # Shared Backbone (Flexible)
        # ----------------------
        if backbone == 'efficientnet':
            self.backbone = tf.keras.applications.EfficientNetV2B0(
                include_top=False,
                input_shape=input_shape,
                weights=None,  # Training from scratch
                pooling='avg'
            )
        elif backbone == 'convnext':
            raise NotImplementedError("ConvNeXt backbone is not yet available in this TensorFlow/Keras setup. Consider using EfficientNet for now.")

            # ConvNeXtV2B0
            self.backbone = tf.keras.applications.ConvNeXtBase(  # Placeholder, validate available models
                    include_top=False,
                    input_shape=input_shape, 
                    weights=None, 
                    pooling='avg'
            )
        elif backbone == 'swin':
            raise NotImplementedError("SwinV2 is not implemented in TensorFlow yet. Future extension planned.")
        
            # Placeholder: TensorFlow doesn't have official Swin, but if using keras_cv or hub:
            from keras_cv.models import SwinTransformerV2Tiny
            self.backbone = keras_cv.models.SwinTransformerV2B0(  # Placeholder, validate
                include_top=False, input_shape=input_shape, weights=None, pooling='avg'
            )
        elif backbone == 'mamba':
            raise NotImplementedError("Mamba model requires PyTorch implementation. Future integration possible.")

            # To be implemented carefully (if TensorFlow version exists or wrapped PyTorch)
            raise NotImplementedError("Mamba backbone requires custom implementation (not available in TF).")
        else:
            raise ValueError("Unsupported backbone selected.")

        # ----------------------
        # Output Heads
        # ----------------------
        self.energy_loss_head = layers.Dense(1, activation='sigmoid', name='energy_loss_output')
        self.alpha_head = layers.Dense(3, activation='softmax', name='alpha_output')
        self.q0_head = layers.Dense(4, activation='softmax', name='q0_output')

    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)

        # Multi-head outputs
        energy_output = self.energy_loss_head(x)
        alpha_output = self.alpha_head(x)
        q0_output = self.q0_head(x)

        # return energy_output, alpha_output, q0_output
        return {
        'energy_loss_output': energy_output,
        'alpha_output': alpha_output,
        'q0_output': q0_output
        }


# ---------------------------
# Model Creation Helper
# ---------------------------

def create_model(backbone='efficientnet', input_shape=(32, 32, 1),learning_rate=0.001):
    """
    Helper to create an instance of MultiHeadClassifier.
    """
    model = MultiHeadClassifier(backbone=backbone, input_shape=input_shape)

    # Compile model with multiple losses (can be adjusted in training script)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'energy_loss_output': tf.keras.losses.BinaryCrossentropy(),
            'alpha_output': tf.keras.losses.CategoricalCrossentropy(),
            'q0_output': tf.keras.losses.CategoricalCrossentropy(),
        },
        metrics={
            'energy_loss_output': ['accuracy'],
            'alpha_output': ['accuracy'],
            'q0_output': ['accuracy'],
        }
    )
    return model


# ---------------------------
# Example Usage (Optional)
# ---------------------------

# if __name__ == "__main__":
#     model = create_model()
#     model.build(input_shape=(None, 32, 32, 1))
#     model.summary()

# Example input to test the model
# python models/model.py
# python models/model.py --backbone efficientnet
# python models/model.py --backbone convnext
# python models/model.py --backbone swin
# python models/model.py --backbone mamba

if __name__ == "__main__":
    import argparse

    # -------------------------------
    # Argument Parser for Backbone Testing
    # -------------------------------
    parser = argparse.ArgumentParser(description="Test different backbone models for ML-JET multi-head classifier.")
    parser.add_argument('--backbone', type=str, default='efficientnet',
                        help='Backbone model to test (options: efficientnet, convnext, swin, mamba)')
    parser.add_argument('--input_shape', type=int, nargs=3, default=[32, 32, 1],
                        help='Input shape as three integers (default: 32 32 1)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    args = parser.parse_args()

    print(f"\n[INFO] Testing backbone: {args.backbone}")
    print(f"[INFO] Input shape: {args.input_shape}")
    print(f"[INFO] Learning rate: {args.learning_rate}")

    # -------------------------------
    # Create and Build Model
    # -------------------------------
    model = create_model(backbone=args.backbone,
                         input_shape=tuple(args.input_shape),
                         learning_rate=args.learning_rate)
    
    model.build(input_shape=(None, *args.input_shape))

    # -------------------------------
    # Print Model Summary
    # -------------------------------
    model.summary()

