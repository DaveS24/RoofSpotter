import tensorflow as tf


class Backbone:
    '''
    The ResNet50 backbone for the Mask R-CNN model.
    
        Attributes:
            config (Config): The configuration settings.
            model (tf.keras.Model): The ResNet50 model.

        Methods:
            build_model: Load a pre-trained ResNet50 model and freeze layers.
    '''

    def __init__(self, config, name='Backbone'):
        self.config = config

        self.model = self.build_model()
        self.model._name = name


    def build_model(self):
        '''
        Load a pre-trained ResNet50 model and freeze layers.

            Parameters:
                None

            Returns:
                model (tf.keras.Model): The ResNet50 model.
        '''

        input_image = tf.keras.layers.Input(shape=self.config.image_shape, batch_size=self.config.batch_size, name='input_image')
        
        # Load a pre-trained ResNet50 model
        resnet50 = tf.keras.applications.ResNet50(input_tensor=input_image,
                                                  weights=self.config.backbone_weights,
                                                  include_top=False)

        # Freeze all layers except the last `trainable_layers`
        for layer in resnet50.layers[:-self.config.backbone_trainable_layers]:
            layer.trainable = False

        model = tf.keras.Model(inputs=input_image, outputs=resnet50.output)
        return model
    