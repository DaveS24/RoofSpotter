# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


## [Unreleased]

### Added

- Set fix batch size for each component.
- Model plot to display the model's architecture.
- Descriptive comments about the output shapes in the rpn.py file.
- The 'alignments.ipynb' file to the project's testing folder.
    - A notebook to test the logic of the ROI Align layer.
- Testing of the sampling-point generation in the alignments.ipynb file.
- Testing of the neighbor-point finding in the alignments.ipynb file.
- Testing of the weights calculation in the alignments.ipynb file.

### Changed

- Model's input dimensions to be fixed for each component.
- Set placeholders for the ROI Align, Classifier, and Mask Head components.
- Data-paths to be set in the Config class in the utils.py file.

### Fixed

- The load_mask() function in the data_loading.py file to only load one color channel.
- The NMS to work correctly for each feature map in the batch.
- The visualizations to work correctly for the fixed batch size.
- Minor rephrasing in multiple files.


## [0.2.11] - 2024-03-18

### Added

- Clip the bounding box predictions to the feature map dimensions in the rpn.py file.
- Documentation in the visualize.py file.
- Documentation in the utils.py file.
- Documentation in the data_loading.py file.
- Documentation in the mask_rcnn.py file.
- Documentation in the backbone.py file.
- Documentation in the rpn.py file.

### Fixed

- Added a fm_offset parameter to correctly visualize the ROI's on the feature map in the visualize.py file.


## [0.2.10] - 2024-03-16

### Added

- batch_size parameter to the Config class in the utils.py file.
- The display_rois() function to visualize the RPN's predicted rois on both the image and feature map in the visualize.py file.
- Set random seeds of the numpy, tensorflow, and random modules to 0 in the mask_rcnn.py file.

### Changed

- The structure of the RPN to work more efficiently and to be more readable.
- The anchor generation to be more efficient and to work with the new RPN.
- Order of operations to decode first and then apply NMS in the rpn.py file.

### Fixed

- The scales and ratios of the anchors to be correctly sized for the feature maps.
- The NMS to work correctly with the shape of the rois.
- The decode_offsets() function in the rpn.py file to correctly decode the bounding box predictions.
- The NMS to receive the coordinates in the format (y1, x1, y2, x2) in the rpn.py file.


## [0.2.9] - 2024-03-12

### Added

- Citation of the dataset to the README.md file.
- Visualization of the backbone's average feature map in the visualize.py file.
- component_testing.ipynb file to the project's testing folder.
    - A notebook to test the individual components of the pipeline.

### Changed

- The wording of the Changelog to be more consistent.
- Refactored the `Config` class to include more settings that are used throughout the components, replacing previously hard-coded values.
- Moved the AnchorGenerator class to the RPN.py file as it is only used there.

### Fixed

- The decode_boxes() function in the rpn.py file to correctly decode the bounding box predictions.


## [0.2.8] - 2024-03-06

### Changed

- Structure of the rpn.py file to improve readability.

### Fixed

- Input dimensions of the RPN in the rpn.py file.
- Input dimensions of the ROI Align layer in the roi_align.py file.
- Aligned feature calculation in the roi_align.py file.
- Output dimensions of the classifier in the classifier.py file.
- Inputs of the mask head in the mask_head.py file.
    (The model builds successfully!)


## [0.2.7] - 2024-03-04

### Fixed

- Dimensions of the anchors in the rpn.py file.
- Dimensions of the rois in the roi_align.py file.
- Dimensions of the region proposals in the roi_align.py file.
- Reduced memory usage in the classifier.py file by implementing a convolutional layer.
- Dimensions of the aligned features in the mask_head.py file.


## [0.2.6] - 2024-03-04

### Added

- Visualization of the model's predictions in the visualize.py file.
    - display_sample() function to display a sample of the dataset (image, mask, and overlay).
    - display_feature_maps() function to display the feature maps of the backbone.

### Removed

- The 'data_tests.ipynb' file from the project's notebooks folder due to merging the tests into the 'pipeline_tests.ipynb' file.


## [0.2.5] - 2024-03-02

### Added

- Loss Function in the mask_head.py file.
    - Mask Loss

### Fixed

- Removed the '.vscode' folder from the project's repository.


## [0.2.4] - 2024-03-02

### Added

- Config class to configure the model's parameters in the utils.py file.

### Fixed

- Excluded the '.vscode' folder from the project's repository.


## [0.2.3] - 2024-03-02

### Added

- Loss Function in the classifier.py file.
    - Classification Loss
    - Regression Loss

### Changed

- Reshape the bounding box predictions into a 3D tensor with shape (batch_size, num_classes, 4) in the classifier.py file.


## [0.2.2] - 2024-03-02

### Added

- ROI Integration in the roi_align.py file.
- Bilinear Interpolation in the roi_align.py file.


## [0.2.1] - 2024-03-02

### Added

- TODO-lists in each component of the pipeline to keep track of the tasks that need to be done.
- Anchor Application and Proposal Generation in the rpn.py file.
    - Sliding Window
    - Anchor Mapping
    - Proposal Generation
- Objectness Scores and Refinement in the rpn.py file.
    - Score Calculation
    - Bounding Box Regression
- Loss Function in the rpn.py file.
    - Classification Loss
    - Regression Loss
- Non-Maximum Suppression in the rpn.py file.


## [0.2.0] - 2024-02-29

### Added

- The data_loading.py file to the project's src folder.
    - The BavarianBuildingDataset class contains the methods to load and split the dataset.
    - A generator() function to load the dataset in batches during training.

- The mask_rcnn.py file to the project's src folder.
    - The MaskRCNN class contains the methods to construct the Mask R-CNN model.

- Tiles for each of the components of the Mask R-CNN model in the src folder.
    - backbone.py contains the ResNet50 backbone.
    - rpn.py contains the Region Proposal Network.
    - roi_align.py contains the Region of Interest Align layer.
    - classifier.py contains the classifier layer.
    - mask_head.py contains the mask head.
    - utils.py contains the utility functions for the model.
    - visualize.py contains the visualization functions for the model.

- The data_tests.ipynb file to the project's notebooks folder.
    - A notebook to test the data_loading.py file.
- The pipeline_tests.ipynb file to the project's notebooks folder.
    - A notebook to test the individual components of the pipeline.


## [0.1.1] - 2024-02-27

### Added

- The 'The Bavarian Buildings Dataset' to the project's data folder.
    - bbd2k5-image: 2500px x 2500px satellite images of Bavaria.
    - bbd2k5-umring: 2500px x 2500px binary masks of the buildings in the images.
    - bbd250-image: 250px x 250px satellite images of Bavaria.
    - bbd250-umring: 250px x 250px binary masks of the buildings in the images.

### Changed

- The project's structure to a more organized one.
- The README.md file with a more detailed project description.


## [0.1.0] - 2024-02-15

### Added

- An empty .gitignore file.

### Changed

- The README.md file with a general project description.
