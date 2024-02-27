# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


## [Unreleased]

### Added

- Added the data_loading.py file to the project's src folder.
    - The BavarianBuildingDataset class contains the methods to load and split the dataset.
    - A generator() method to load the dataset in batches during training.
- Added the data_analysis.ipynb file to the project's notebooks folder.
    - The file contains the code to analyze the dataset.


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
- Decided to use 'The Bavarian Buildings Dataset' as the dataset for this project.

## [0.1.0] - 2024-02-15

### Added

- An empty .gitignore file.

### Changed

- Updated the README.md file with a general project description.