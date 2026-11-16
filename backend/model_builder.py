"""
Deep Learning Model Builder
Defines deep neural network architectures for Sign Language Recognition.
"""

import keras
from keras import layers, models, regularizers


def build_sign_classifier(input_dim=76, num_classes=19, learning_rate=0.001):
    """
    Constructs a Deep Multilayer Perceptron with Batch Normalization
    and Dropout regularizations for robust sign classification.
    """
    inputs = layers.Input(shape=(input_dim,), name="landmarks_input")

    # Layer Block 1
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), name="dense_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.Dropout(0.3, name="drop_1")(x)

    # Layer Block 2
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4), name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.Dropout(0.25, name="drop_2")(x)

    # Layer Block 3
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4), name="dense_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Activation("relu", name="relu_3")(x)
    x = layers.Dropout(0.2, name="drop_3")(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation="softmax", name="sign_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SignFlow_Classifier")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

