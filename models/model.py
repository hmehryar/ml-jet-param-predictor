# model.py

import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------------------------
# Multi-Head Classifier Model
# ---------------------------

class MultiHeadClassifier(Model):
    def __init__(self, backbone='efficientnet', input_shape=(32, 32, 1)):
        super(MultiHeadClassifier, self).__init__()

        # ----------------------
        # Shared Backbone
        # ----------------------
        if backbone == 'efficientnet':
            self.backbone = tf.keras.applications.EfficientNetV2B0(
                include_top=False,
                input_shape=input_shape,
                weights=None,  # Training from scratch
                pooling='avg'
            )
        elif backbone == 'convnext':
            # Example placeholder for ConvNeXt if added later
            raise NotImplementedError("ConvNeXt backbone not implemented yet.")
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

def create_model(backbone='efficientnet', input_shape=(32, 32, 1)):
    """
    Helper to create an instance of MultiHeadClassifier.
    """
    model = MultiHeadClassifier(backbone=backbone, input_shape=input_shape)

    # Compile model with multiple losses (can be adjusted in training script)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
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

if __name__ == "__main__":
    model = create_model()
    model.build(input_shape=(None, 32, 32, 1))
    model.summary()

# Example input to test the model
# python models/model.py
