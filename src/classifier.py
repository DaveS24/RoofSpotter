import tensorflow as tf


class Classifier:
    '''
    The classifier for the Mask R-CNN model.

        Attributes:
            config (Config): The configuration settings.
            roi_align (ROIAlignLayer): The ROI Align layer for the Mask R-CNN model.
            model (tf.keras.Model): The classifier for the Mask R-CNN model.
            
        Methods:
            build_model: Link the components to build the classifier.
    '''

    def __init__(self, config, roi_align, name='Classifier'):
        self.config = config
        self.roi_align = roi_align

        self.model = self.build_model()
        self.model._name = name


    def build_model(self):
        '''
        Link the components to build the classifier.

            Parameters:
                None
                
            Returns:
                model (tf.keras.Model): The classifier for the Mask R-CNN model.
        '''

        # Get the aligned ROIs from the ROI Align layer
        roi_aligned = self.roi_align.model.output

        x = tf.keras.layers.Flatten()(roi_aligned)

        # Apply a fully connected layer to extract features
        x = tf.keras.layers.Dense(self.config.classifier_dense_units, activation='relu')(x)

        # Apply a fully connected layer to predict the class scores
        class_scores = tf.keras.layers.Dense(self.config.num_classes, activation='softmax', name='class_scores')(x)

        # Apply a fully connected layer to predict the bounding box coordinates
        bbox = tf.keras.layers.Dense(self.config.num_classes * 4, activation='linear', name='bbox')(x)
        bbox = tf.keras.layers.Reshape((self.config.num_classes, 4))(bbox)

        model = tf.keras.Model(inputs=roi_aligned, outputs=[class_scores, bbox])
        return model
    