# Machine Learning Framework for analyzing Satellite Data & Image Classification of Transmission Lines
#### Description
This project outlines a framework for Integrating Deep Learning Models for Spatial Data Analysis with a focus on Image Classification for Power Grid Lines in Satellite Images.

## PART 1: The Framework
“Satellite images contain an incredible amount of data about the world, but the trick is how to translate the data into usable insights without having a human comb through every single image,” - Esther Rolf

#### Goal:
The HV grid is always in flux, so the ability to create an accurate snapshot at regular intervals is an important tool. The follwoing framework uses neural networks to train machine learning models to analyse satellite images and/or spatial data. 

#### Required Tools and Functionalities:
##### Tensorflow:
To train the deep learning models, we will need to import various libraries and modules in Python. Tensorflow and PyTorch are the two most commonly used libraries. For this project, the open-source deep learning library Tensoflow, created by the Google Brain team, has been used. 

One of the key features of TensorFlow is it's ability to automatically compute gradients, making it suitable for implementing gradient-based optimization algorithms for training neural networks through backpropagation. This feature simplifies the process of building and training complex models. It also has a wide range of pre-built models and tools, making it suitable for various domains, like computer vision, natural language processing, etc. 

##### Keras:
TensorFlow provides a high-level API called Keras as `tf.keras`, which offers an intuitive and user-friendly interface for building neural networks and other machine learning models

##### Google Colab:
Google Colab (short for Google Colaboratory) provides a cloud-based Jupyter notebook environment with access to GPUs (Graphical Processing Unit) and TPUs (Tensor Processing Unit) enabling users to accelerate computations, making it ideal for training deep learning models efficiently. Colab also offers a seamless integration with Google Drive

#### Dataset:
The first step for preparing a dataset is to find either a public dataset or a data source to collect the required data from. If the data is being collected from a source, the right data format, dimension, and label has to be maintained. The next is organizing the data into folders and subfolders and storing the customized dataset in a defined path. Each of the folders inside the main folder should represent a separate yclass or category. For example, if you are building a cat vs. dog classifier, you can have two folders: "cat" and "dog", with respective images inside each folder.

Tensorflow supports various image types

#### Public Datasets:

#### Models and Hyperparameters:

#### Transfer Learning & Pre-trained Models

#### Scoring & Visualization Mechanisms:

#### Auto-ML

* explore TIFF 
Abstract: Use ML and/or DL models on the training set of annotated and labeled data of the missing transmission lines (above 230kV) as compared to HIFLD and test it on the other transmission lines of lower V.

new from the HIFLD data - compare to HE TL layer - get unmatched -  find raster images for locations - analysis 

Training/Validation Dataset:

Test Dataset:

Data Format:
Tab, shp, raster, GeoJSON

Models & Hyperparameters:

Scoring Mechanisms: 
the widely used metric for evaluating object detection performance is mAP (mean Average Precision). mAP measures the precision and recall of object detection algorithms across different object categories and provides an overall performance score.

Public Datasets: 

1. SpaceNet: SpaceNet is a series of open-source satellite imagery datasets focused on various geospatial challenges. While not specifically targeting power grid lines, it includes high-resolution satellite imagery that might contain power grid infrastructure.
2. OpenStreetMap (OSM): OSM is a collaborative mapping project that allows users to contribute and access geospatial data. Power grid lines are sometimes mapped in OSM, and you can extract aerial imagery from various sources associated with OSM data.
3. National Renewable Energy Laboratory (NREL)? datasets: NREL provides a collection of geospatial datasets related to renewable energy, including solar and wind energy. These datasets often encompass aerial imagery that may include power grid lines in proximity to renewable energy installations.
4. Govt. Datasets: HIFLD
5. DeepGlobe : the DeepGlobe dataset does not specifically include power grid lines as a labeled category. The DeepGlobe dataset focuses on several computer vision tasks related to satellite imagery, such as road extraction, building detection, and land cover classification. However, it does not provide annotations or labels specifically for power grid lines.
6. PG imagery dataset:

Transfer Learning Resources:
1. ResNet: ResNet (Residual Neural Network) is a widely used deep convolutional neural network architecture known for its depth and skip connections. It has shown excellent performance in various computer vision tasks and can serve as a strong backbone for power grid line identification.
2. EfficientNet: EfficientNet is a family of efficient convolutional neural network architectures that achieve state-of-the-art performance while maintaining computational efficiency. It provides a good trade-off between accuracy and computational resources, making it suitable for power grid line identification.
3. VGGNet: VGGNet is a classic deep CNN architecture known for its simplicity and uniformity. While it may be less computationally efficient compared to newer models, it can still be effective for transfer learning and power grid line identification tasks.
4. Mask R-CNN: Mask R-CNN is a popular instance segmentation framework that combines object detection and pixel-wise segmentation. It can be applied to identify power grid lines by segmenting the lines from the background, providing more detailed information about their locations.
5. YOLO (You Only Look Once): YOLO is an object detection framework that focuses on real-time performance. It can be used to detect power grid lines as bounding boxes with class labels, making it suitable for applications where real-time processing is crucial.

## check out road - mapping?

How to utilize transfer learning? --- 
--- fine-tune the pre-trained models on a specific power grid line dataset to adapt them to the target task.
--- freeze certain layers, replace or add new layers, and retrain the model on the power grid line data to achieve better performance.
![image](https://github.com/farastu-who/GIS-DL/assets/34352153/456713f7-f574-4565-add5-a2e6ee03f434)

## PART 2: TLC (Tranmission Line Classification) - Image Classification of Transmission Lines using Satellite Data

#### Required Tools and Functionalities:

#### Dataset:

#### Public Datasets:

#### Models and Hyperparameters:

#### Transfer Learning & Pre-trained Models

#### Scoring & Visualization Mechanisms:

#### Auto-ML

#### Further Work: 
-make greyscale
- more data
- labelled data
- models
- object detection
#### Resources
1. https://medium.com/spatial-data-science/deep-learning-for-geospatial-data-applications-multi-label-classification-2b0a1838fcf3#:~:text=In%20this%20tutorial%2C%20I%20will%20show%20the%20easiest,lines%20of%20python%20code%20to%20accomplish%20this%20task.
e
2. https://scholar.google.com/citations?user=n1EE3-8AAAAJ
3. 

power line image dataset that are publicly available:
1. TTPLA -
2. Emre, Y.Ö., Nezih, G.Ö., et al.: Power line image dataset (infrared-IR and visible
light-VL). Mendeley Data (2017)
3. 
