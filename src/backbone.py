import tensorflow as tf


class Backbone:
    def __init__(self, input_shape=(250, 250, 3), trainable_layers=3):
        self.input_shape = input_shape
        self.trainable_layers = trainable_layers
        self.model = self.build_model()

    def build_model(self):
        # Load a pre-trained ResNet50 model
        resnet50 = tf.keras.applications.ResNet50(
            weights='imagenet', include_top=False, input_shape=self.input_shape
        )

        # Freeze all layers except the last `trainable_layers`
        for layer in resnet50.layers[:-self.trainable_layers]:
            layer.trainable = False

        # Create a new model with the desired input and output
        model = tf.keras.Model(
            inputs=resnet50.inputs, outputs=resnet50.output
        )
        return model
    