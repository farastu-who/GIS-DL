# Machine Learning Framework for analyzing Satellite Data & Image Classification of Transmission Lines
#### Description
This project outlines a framework for Integrating Deep Learning Models for Spatial Data Analysis with a focus on Image Classification for Power Grid Lines in Satellite Images.

## PART 1: The Framework
“Satellite images contain an incredible amount of data about the world, but the trick is how to translate the data into usable insights without having a human comb through every single image,” - Esther Rolf

#### a) Goal:
The HV grid is always in flux, so the ability to create an accurate snapshot at regular intervals is an important tool. The following framework uses neural networks to train machine learning models to analyze satellite images and/or spatial data. 

#### b) Required Tools and Functionalities:

##### Google Colab:
Google Colab (short for Google Colaboratory) provides a cloud-based Jupyter notebook environment with access to GPUs (Graphical Processing Units) and TPUs (Tensor Processing Units), enabling users to accelerate computations, making it ideal for training deep learning models efficiently. Colab also offers seamless integration with Google Drive.

##### Tensorflow:
To train the deep learning models, we will need to import various libraries and modules in Python. Tensorflow and PyTorch are the two most commonly used libraries. For this project, the open-source deep learning library TensorFlow, created by the Google Brain team, has been used. 

One of the key features of TensorFlow is its ability to automatically compute gradients, making it suitable for implementing gradient-based optimization algorithms for training neural networks through backpropagation. This feature simplifies the process of building and training complex models. It also has a wide range of pre-built models and tools, making it suitable for various domains, like computer vision, natural language processing, etc. 

##### Keras:
TensorFlow provides a high-level API called Keras as `tf.keras`, which offers an intuitive and user-friendly interface for building neural networks and other machine learning models.

#### c) Dataset:
The first step for preparing a dataset is to find either a public dataset or a data source to collect the required data. If the data is being collected from a source, the right data format, dimension, and labels have to be maintained. The next is organizing the data into folders and subfolders and storing the customized dataset in a defined path. Each of the folders inside the main folder should represent a separate class or category. For example, if you are building a cat vs. dog classifier, you can have two folders: "cat" and "dog", with respective images inside each folder.

Tensorflow supports various image types:
JPEG (Joint Photographic Experts Group), PNG (Portable Network Graphics), GIF (Graphics Interchange Format), BMP (Bitmap), TIFF (Tagged Image File Format), WebP

A few of the commonly used datasets for aerial imagery are outlined below:

1. SpaceNet: SpaceNet is a series of open-source satellite imagery datasets focused on various geospatial challenges. 
2. OpenStreetMap (OSM): OSM is a collaborative mapping project that allows users to contribute and access geospatial data. Power grid lines are sometimes mapped in OSM, and you can extract aerial imagery from various sources associated with OSM data.
3. National Renewable Energy Laboratory (NREL) datasets: NREL provides a collection of geospatial datasets related to renewable energy, including solar and wind energy. These datasets often encompass aerial imagery that may include power grid lines in proximity to renewable energy installations.
4. DeepGlobe: The DeepGlobe dataset focuses on several computer vision tasks related to satellite imagery, such as road extraction, building detection, and land cover classification. However, it does not provide annotations or labels specifically for power grid lines.
5. TTPLA: This is a public dataset that is a collection of aerial images of Transmission Towers (TTs) and Powers Lines (PLs). An example from the dataset is shown below.

![TTPLA dataset](https://github.com/farastu-who/GIS-DL/assets/34352153/1fdf015b-2baf-4179-bfbf-eeac545c3504)

For object detection, the images need to be annotated with bounding boxes, and the following steps are needed:

1. Image Annotation: It involves manually or semi-automatically labeling objects of interest within an image with corresponding annotations. These annotations provide ground truth information that helps train object detection models to identify and localize objects in new, unseen images.
2. Bounding Boxes: A bounding box is a rectangular region defined by four points: the coordinates of the top-left corner (x, y) and the bottom-right corner (x+w, y+h), where w and h are the width and height of the box, respectively. The bounding box surrounds the object of interest in the image. Each bounding box is associated with a specific class label that identifies the type of object it encloses.
3. Labeling Objects: During the annotation process, an annotator manually draws bounding boxes around the objects in the image using specialized annotation tools or software. The annotator also assigns class labels to each bounding box, indicating the type of object it represents (e.g., car, person, dog, etc.).

![elephant](https://github.com/farastu-who/GIS-DL/assets/34352153/dbcf6eff-3312-44f3-a30f-0e940b4b17f7)

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

An explanation of a basic CNN architecture model, along with the commonly used hyperparameters, have been delineated below:

##### CNN Architecture Model:

1. Input Layer: The input layer receives the raw pixel values of an image as its input. Images are usually represented as 3D tensors with dimensions (height, width, channels), where channels represent color channels. For example, if the images are grayscale with a size of 28x28 pixels, the input_shape would be (28, 28, 1). If the images are RGB with a size of 32x32 pixels, the input_shape would be (32, 32, 3). 
   
2. Convolutional Layers: The core building blocks of a CNN are the convolutional layers. Each convolutional layer consists of multiple filters (kernels) that slide over the input image to perform a convolution operation. The filters learn to extract local patterns and features from the input image. The output of each convolutional layer is called a feature map, representing the activations of different filters.
`Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)`
In the above example of a convolutional layer, the Conv2D layer has 32 filters of size 3x3

3. Activation Functions (ReLU): After each convolution operation, an activation function is applied element-wise to introduce non-linearity to the model. The Rectified Linear Unit (ReLU) is commonly used as the activation function in CNNs, setting negative values to zero and keeping positive values unchanged.

4. Pooling Layers: Pooling layers downsample the feature maps obtained from the convolutional layers by reducing their spatial dimensions. MaxPooling and AveragePooling are popular pooling methods, which take the maximum or average value in a local region, respectively.
Pooling reduces computational complexity, helps in controlling overfitting, and makes the model more robust to small variations in the input.

5. Flatten Layer: After the last pooling layer, a flatten layer is used to convert the 3D feature maps into a 1D vector. This step is necessary to connect the output of the convolutional and pooling layers to the fully connected layers (dense layers).

6. Fully Connected Layers: The fully connected layers are traditional neural network layers where every neuron is connected to every neuron in the previous and subsequent layers. These layers process the extracted features and produce the final output for the given task. In Keras, the Dense layer represents a fully connected layer in a neural network.

7. Output Layer: The output layer of the CNN produces the final prediction or classification. For image classification tasks, it typically contains neurons corresponding to the number of classes in the dataset and the output is passed through an activation function (e.g., softmax for multi-class classification) to obtain class probabilities.

The following is an example of a fully connected layer and the output layer:
```
Dense(128, activation='relu'),
Dense(num_classes, activation='softmax')
```
The Dense layer constructor accepts two main parameters; units and activation_function. Units are the number of neurons in the layer. In the first Dense layer, there are 128 neurons, and in the second Dense layer, the number of neurons is equal to the number of classes (num_classes) in the dataset.

##### Hyperparameters:

Hyperparameters are parameters that are set before training the model and control various aspects of the learning process. Some common hyperparameters in a CNN include:

1. Number of Convolutional Layers and Filters: The number of convolutional layers and the number of filters in each layer determine the depth and complexity of the CNN architecture. Deeper networks can learn more complex features but may require more computational resources.

2. Kernel Size: The size of the kernels (filters) used in the convolutional layers. Common kernel sizes are 3x3, 5x5, and 7x7.

3. Pooling Size: The size of the pooling windows used in the pooling layers. Common pooling sizes are 2x2 and 3x3.

4. Stride: The step size of the filter as it slides over the input image during convolution. A larger stride reduces the size of the output feature map.

5. Padding: Padding adds extra border pixels to the input image to ensure that the convolution operation does not shrink the spatial dimensions too much. It can be "valid" (no padding) or "same" (pad to retain the spatial dimensions).

6. Activation Function: The choice of activation function, commonly ReLU, but other functions like Sigmoid and Tanh can be used in certain scenarios.

7. Number of Fully Connected Layers and Neurons: The number of fully connected layers and the number of neurons in each layer determine the depth and capacity of the fully connected part of the CNN.

8. Learning Rate: The learning rate controls the step size during gradient descent optimization. It determines how much the model's weights are updated during training.

9. Batch Size: The number of samples used in each training iteration. Larger batch sizes can speed up training but may require more memory.

10. Number of Epochs: The number of times the model goes through the entire training dataset during training.

These hyperparameters are crucial for building an effective CNN architecture and are often tuned through experimentation to achieve the best performance on a specific task and dataset. Different combinations of hyperparameters can significantly impact the model's training time, convergence, and generalization ability.


Overfitting:



#### e) Transfer Learning & Pre-trained Models

A deep learning approach called transfer learning uses a model that has been trained on one problem as the foundation for learning how to solve related problems. Pre-trained models created for benchmark datasets like ImageNet can be utilized again in computer vision applications to reduce training time and improve performance.

The procedure entails incorporating one or more layers from a pre-trained model into a new model that has been trained on the particular topic of interest. This may be accomplished in a number of ways, including initializing the weights of the new model based on the pre-trained model or utilizing the pre-trained model as a feature extractor.

Transfer learning is useful because it enables us to take the skills we've developed for one problem—like distinguishing between cats and dogs—and apply them to another—like distinguishing between ants and wasps—even when the target categories are different. We may make use of models' capacity to recognize generic characteristics in images and obtain state-of-the-art performance by utilizing models that have been trained on huge datasets with various categories.

Many of the best-performing models, including as `VGG`, `Inception`, and `ResNet`, which were trained on the `ImageNet` dataset, are readily accessible through Keras and other deep-learning tools.

<img width="768" alt="image" src="https://github.com/farastu-who/GIS-DL/assets/34352153/3aa41ac0-537b-4fe8-94ff-4402f3f9786e">

#### f) Scoring & Visualization Mechanisms:

#### g) Auto-ML

 
Abstract: Use ML and/or DL models on the training set of annotated and labeled data of the missing transmission lines (above 230kV) as compared to HIFLD and test it on the other transmission lines of lower V.

new from the HIFLD data - compare to HE TL layer - get unmatched -  find raster images for locations - analysis 

#### h) Inference Integration:

#### i Challenges:

1. Overfitting
2. Underfitting
3. Hyperparameter Tuning
4. Data Quality and Quantity
5. Imbalanced Data
6. Computational Resources



Data Format:
Tab, shp, raster, GeoJSON



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

## PART 2: TLC (Transmission Line Classification) - Image Classification of Transmission Lines using Satellite Data

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
- labeled data
- models
- object detection
- explore TIFF
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
