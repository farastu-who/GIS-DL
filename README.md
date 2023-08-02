# Machine Learning Framework for analyzing Satellite Data & Image Classification of Transmission Lines
#### Description
This project outlines a framework for Integrating Deep Learning Models for Spatial Data Analysis with a focus on Image Classification for Power Grid Lines in Satellite Images.

## PART 1: The Framework
“Satellite images contain an incredible amount of data about the world, but the trick is how to translate the data into usable insights without having a human comb through every single image,” - Esther Rolf

#### a) Goal:
The HV grid is always in flux, so the ability to create an accurate snapshot at regular intervals is an important tool. The follwoing framework uses neural networks to train machine learning models to analyse satellite images and/or spatial data. 

#### b) Required Tools and Functionalities:

##### Google Colab:
Google Colab (short for Google Colaboratory) provides a cloud-based Jupyter notebook environment with access to GPUs (Graphical Processing Unit) and TPUs (Tensor Processing Unit) enabling users to accelerate computations, making it ideal for training deep learning models efficiently. Colab also offers a seamless integration with Google Drive

##### Tensorflow:
To train the deep learning models, we will need to import various libraries and modules in Python. Tensorflow and PyTorch are the two most commonly used libraries. For this project, the open-source deep learning library Tensoflow, created by the Google Brain team, has been used. 

One of the key features of TensorFlow is it's ability to automatically compute gradients, making it suitable for implementing gradient-based optimization algorithms for training neural networks through backpropagation. This feature simplifies the process of building and training complex models. It also has a wide range of pre-built models and tools, making it suitable for various domains, like computer vision, natural language processing, etc. 

##### Keras:
TensorFlow provides a high-level API called Keras as `tf.keras`, which offers an intuitive and user-friendly interface for building neural networks and other machine learning models

#### c) Dataset:
The first step for preparing a dataset is to find either a public dataset or a data source to collect the required data from. If the data is being collected from a source, the right data format, dimension, and label has to be maintained. The next is organizing the data into folders and subfolders and storing the customized dataset in a defined path. Each of the folders inside the main folder should represent a separate class or category. For example, if you are building a cat vs. dog classifier, you can have two folders: "cat" and "dog", with respective images inside each folder.

Tensorflow supports various image types:
JPEG (Joint Photographic Experts Group), PNG (Portable Network Graphics), GIF (Graphics Interchange Format), BMP (Bitmap), TIFF (Tagged Image File Format), WebP

A few of the commonly used datasets for aerial imagery are outlined below:

1. SpaceNet: SpaceNet is a series of open-source satellite imagery datasets focused on various geospatial challenges. 
2. OpenStreetMap (OSM): OSM is a collaborative mapping project that allows users to contribute and access geospatial data. Power grid lines are sometimes mapped in OSM, and you can extract aerial imagery from various sources associated with OSM data.
3. National Renewable Energy Laboratory (NREL) datasets: NREL provides a collection of geospatial datasets related to renewable energy, including solar and wind energy. These datasets often encompass aerial imagery that may include power grid lines in proximity to renewable energy installations.
4. DeepGlobe : The DeepGlobe dataset focuses on several computer vision tasks related to satellite imagery, such as road extraction, building detection, and land cover classification. However, it does not provide annotations or labels specifically for power grid lines.
5. TTPLA: This is a public dataset which is a collection of aerial images on Transmission Towers (TTs) and Powers Lines (PLs)

For object detection, the images need to be annoted with bounding boxes and the following steps are needed:

1. Image Annotation: It involves manually or semi-automatically labeling objects of interest within an image with corresponding annotations. These annotations provide ground truth information that helps train object detection models to identify and localize objects in new, unseen images.
2. Bounding Boxes:A bounding box is a rectangular region defined by four points: the coordinates of the top-left corner (x, y) and the bottom-right corner (x+w, y+h), where w and h are the width and height of the box, respectively. The bounding box surrounds the object of interest in the image. Each bounding box is associated with a specific class label that identifies the type of object it encloses.
3. Labeling Objects:During the annotation process, an annotator manually draws bounding boxes around the objects in the image using specialized annotation tools or software. The annotator also assigns class labels to each bounding box, indicating the type of object it represents (e.g., car, person, dog, etc.).

Data Generators
```
# Training data generator
train_generator = datagen.flow_from_directory(
    'path_to_train_folder',
    target_size=(height, width),  # Resize images to a specified height and width
    batch_size=batch_size,
    class_mode='categorical',  # For multi-class classification
    subset='training',  # Use the training subset
)
# Validation data generator
validation_generator = datagen.flow_from_directory(
    'path_to_train_folder',
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # Use the validation subset
)

```
Input Shape & Number of Classes
   
#### d) Models and Hyperparameters:

A Convolutional Neural Network (CNN) is a type of deep learning model commonly used for image recognition, computer vision tasks, and other pattern recognition problems. CNNs are designed to automatically and adaptively learn spatial hierarchies of features from input images, making them particularly effective in capturing local patterns and structures.

An explanation of a basic CNN architecture model along with the commonly used hyperparameters have been delineated below:

##### CNN Architecture Model:

1. Input Layer:The input layer receives the raw pixel values of an image as its input. Images are usually represented as 3D tensors with dimensions (height, width, channels), where channels represent color channels (e.g., RGB images have 3 channels).
   
2. Convolutional Layers:The core building blocks of a CNN are the convolutional layers. Each convolutional layer consists of multiple filters (kernels) that slide over the input image to perform a convolution operation. The filters learn to extract local patterns and features from the input image.
The output of each convolutional layer is called a feature map, representing the activations of different filters.
Activation Functions (ReLU):After each convolution operation, an activation function is applied element-wise to introduce non-linearity to the model. The Rectified Linear Unit (ReLU) is commonly used as the activation function in CNNs, setting negative values to zero and keeping positive values unchanged.
Pooling Layers:Pooling layers downsample the feature maps obtained from the convolutional layers by reducing their spatial dimensions. MaxPooling and AveragePooling are popular pooling methods, which take the maximum or average value in a local region, respectively.
Pooling reduces computational complexity, helps in controlling overfitting, and makes the model more robust to small variations in the input.
Flatten Layer:After the last pooling layer, a flatten layer is used to convert the 3D feature maps into a 1D vector. This step is necessary to connect the output of the convolutional and pooling layers to the fully connected layers (dense layers).
Fully Connected Layers:The fully connected layers are traditional neural network layers where every neuron is connected to every neuron in the previous and subsequent layers. These layers process the extracted features and produce the final output for the given task.
Output Layer:The output layer of the CNN produces the final prediction or classification. For image classification tasks, it typically contains neurons corresponding to the number of classes in the dataset, and the output is passed through an activation function (e.g., softmax for multi-class classification) to obtain class probabilities.

Hyperparameters:

Hyperparameters are parameters that are set before training the model and control various aspects of the learning process. Some common hyperparameters in a CNN include:

Number of Convolutional Layers and Filters:The number of convolutional layers and the number of filters in each layer determine the depth and complexity of the CNN architecture. Deeper networks can learn more complex features but may require more computational resources.
Kernel Size:The size of the kernels (filters) used in the convolutional layers. Common kernel sizes are 3x3, 5x5, and 7x7.
Pooling Size:The size of the pooling windows used in the pooling layers. Common pooling sizes are 2x2 and 3x3.
Stride:The step size of the filter as it slides over the input image during convolution. A larger stride reduces the size of the output feature map.
Padding:Padding adds extra border pixels to the input image to ensure that the convolution operation does not shrink the spatial dimensions too much. It can be "valid" (no padding) or "same" (pad to retain the spatial dimensions).
Activation Function:The choice of activation function, commonly ReLU, but other functions like Sigmoid and Tanh can be used in certain scenarios.
Number of Fully Connected Layers and Neurons:The number of fully connected layers and the number of neurons in each layer determine the depth and capacity of the fully connected part of the CNN.
Learning Rate:The learning rate controls the step size during gradient descent optimization. It determines how much the model's weights are updated during training.
Batch Size:The number of samples used in each training iteration. Larger batch sizes can speed up training but may require more memory.
Number of Epochs:The number of times the model goes through the entire training dataset during training.

These hyperparameters are crucial for building an effective CNN architecture and are often tuned through experimentation to achieve the best performance on a specific task and dataset. Different combinations of hyperparameters can significantly impact the model's training time, convergence, and generalization ability.

#### e) Transfer Learning & Pre-trained Models

#### f) Scoring & Visualization Mechanisms:

#### g) Auto-ML

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
