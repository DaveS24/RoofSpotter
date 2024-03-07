# RoofSpotter: Automated Roof Detection Using Mask R-CNN and Building Boundary Regularization

RoofSpotter is an advanced machine learning and deep learning project aimed at automating the detection of house rooftops in satellite images, specifically tailored for the Bavarian region of Germany. Leveraging "The Bavarian Buildings Dataset (BBD)," which provides detailed building footprint information combined with high-resolution imagery, RoofSpotter employs state-of-the-art computer vision techniques to precisely identify and outline rooftop structures.

The project utilizes the Mask R-CNN (Region-based Convolutional Neural Network) architecture as the primary tool for initial rooftop detection. Mask R-CNN is a robust deep learning model capable of simultaneously detecting object instances and generating pixel-wise segmentation masks, making it well-suited for tasks requiring precise spatial localization.

Following the initial detection with Mask R-CNN, RoofSpotter incorporates Building Boundary Regularization techniques to refine and simplify the shape of the detected roofs. This step enhances the accuracy and usability of the detected rooftop outlines by smoothing irregularities and ensuring geometric consistency.

**Dataset Citation**

Werner, M., Li, H., Zollner, J.M., Teuscher, B., & Deuser, F. (2023). Bavaria Buildings - A Novel Dataset for Building Footprint Extraction, Instance Segmentation, and Data Quality Estimation (Data and Resources Paper).  In The 31st ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL 23), November 1316, 2023, Hamburg, Germany. ACM, New York, NY, USA, 4 pages, https://doi.org/10.1145/3589132.3625658.