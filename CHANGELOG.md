# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


## [Unreleased]

### Added

- TODO-lists in each component of the pipeline to keep track of the tasks that need to be done.


## [0.2.0] - 2024-02-29

### Added

- Added the data_loading.py file to the project's src folder.
    - The BavarianBuildingDataset class contains the methods to load and split the dataset.
    - A generator() function to load the dataset in batches during training.

- Added the mask_rcnn.py file to the project's src folder.
    - The MaskRCNN class contains the methods to construct the Mask R-CNN model.

- Added files for each of the components of the Mask R-CNN model in the src folder.
    - backbone.py contains the ResNet50 backbone.
    - rpn.py contains the Region Proposal Network.
    - roi_align.py contains the Region of Interest Align layer.
    - classifier.py contains the classifier layer.
    - mask_head.py contains the mask head.
    - utils.py contains the utility functions for the model.
    - visualize.py contains the visualization functions for the model.

- Added the data_tests.ipynb file to the project's notebooks folder.
    - A notebook to test the data_loading.py file.
- Added the pipeline_tests.ipynb file to the project's notebooks folder.
    - A notebook to test the individual components of the pipeline.


## [0.1.1] - 2024-02-27

### Added

- Added the 'The Bavarian Buildings Dataset' to the project's data folder.
    - bbd2k5-image: 2500px x 2500px satellite images of Bavaria.
    - bbd2k5-umring: 2500px x 2500px binary masks of the buildings in the images.
    - bbd250-image: 250px x 250px satellite images of Bavaria.
    - bbd250-umring: 250px x 250px binary masks of the buildings in the images.

### Changed

- Changed the project's structure to a more organized one.
- Updated the README.md file with a more detailed project description.


## [0.1.0] - 2024-02-15

### Added

- An empty .gitignore file.

### Changed

- Updated the README.md file with a general project description.
