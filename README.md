# Machine Learning Framework for analyzing Satellite Data & Image Classification of Transmission Lines


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#description">Description</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#part-1-the-framework">PART 1: The Framework</a>
      <ul>
        <li><a href="#goal">Goal</a></li>
        <li><a href="#required-tools-and-functionalities">Required Tools and Functionalities</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- Description -->
### Description
This project outlines a framework for Integrating Deep Learning Models for Spatial Data Analysis with a focus on Image Classification for Power Grid Lines in Satellite Images.

<!-- PART 1: The Framework -->
## PART 1: The Framework
“Satellite images contain an incredible amount of data about the world, but the trick is how to translate the data into usable insights without having a human comb through every single image,” - Esther Rolf

![image](https://github.com/farastu-who/GIS-DL/assets/34352153/d8567584-cc04-4485-a869-ecefa966b863)


<!-- Goal -->
### a) Goal:
The HV grid is always in flux, so the ability to create an accurate snapshot at regular intervals is an important tool. The following framework uses neural networks to train machine learning models to analyze satellite images and/or spatial data. 

<!-- Required Tools and Functionalities -->
### b) Required Tools and Functionalities:

#### Python:
Python is the preferred language for any machine learning and data processing tasks due to it's vast ecosystem, community support, simplicity, flexibility, a plethora of machine learning frameowrks and pre-trained models, and its ease of integration with other data science tools.

#### Google Colab:
Google Colab (short for Google Colaboratory) provides a cloud-based Jupyter notebook environment with access to GPUs (Graphical Processing Units) and TPUs (Tensor Processing Units), enabling users to accelerate computations, making it ideal for training deep learning models efficiently. Colab also offers seamless integration with Google Drive.

#### Tensorflow:
To train the deep learning models, we will need to import various libraries and modules in Python. Tensorflow and PyTorch are the two most commonly used libraries. For this project, the open-source deep learning library TensorFlow, created by the Google Brain team, has been used. 

One of the key features of TensorFlow is its ability to automatically compute gradients, making it suitable for implementing gradient-based optimization algorithms for training neural networks through backpropagation. This feature simplifies the process of building and training complex models. It also has a wide range of pre-built models and tools, making it suitable for various domains, like computer vision, natural language processing, etc. 

#### Keras:
TensorFlow provides a high-level API called Keras as `tf.keras`, which offers an intuitive and user-friendly interface for building neural networks and other machine learning models.

<img width="737" alt="image" src="https://github.com/farastu-who/GIS-DL/assets/34352153/1367fa15-fca6-4805-af71-44905334ca7d">


### c) Dataset:
The first step for preparing a dataset is to find either a public dataset or a data source to collect the required data. If the data is being collected from a source, the right data format, dimension, and labels have to be maintained. The next is organizing the data into folders and subfolders and storing the customized dataset in a defined path. Each of the folders inside the main folder should represent a separate class or category. For example, if you are building a cat vs. dog classifier, you can have two folders: "cat" and "dog", with respective images inside each folder.

Tensorflow supports various image types:
JPEG (Joint Photographic Experts Group), PNG (Portable Network Graphics), GIF (Graphics Interchange Format), BMP (Bitmap), TIFF (Tagged Image File Format), WebP

A few of the commonly used datasets for aerial imagery are outlined below:

1. SpaceNet: SpaceNet is a series of open-source satellite imagery datasets focused on various geospatial challenges. 
2. OpenStreetMap (OSM): OSM is a collaborative mapping project that allows users to contribute and access geospatial data. Power grid lines are sometimes mapped in OSM, and you can extract aerial imagery from various sources associated with OSM data.
3. National Renewable Energy Laboratory (NREL) datasets: NREL provides a collection of geospatial datasets related to renewable energy, including solar and wind energy. These datasets often encompass aerial imagery that may include power grid lines in proximity to renewable energy installations.
4. DeepGlobe: The DeepGlobe dataset focuses on several computer vision tasks related to satellite imagery, such as road extraction, building detection, and land cover classification. However, it does not provide annotations or labels specifically for power grid lines.
5. TTPLA: This is a public dataset that is a collection of aerial images of Transmission Towers (TTs) and Powers Lines (PLs). It consists of 1,100 images with a resolution of 3,840×2,160 pixels, as well as manually labeled 8,987 instances of TTs and PLs. An example from the dataset is shown below.

![TTPLA dataset](https://github.com/farastu-who/GIS-DL/assets/34352153/1fdf015b-2baf-4179-bfbf-eeac545c3504)

For object detection, the images need to be annotated with bounding boxes, and the following steps are needed:

1. Image Annotation: It involves manually or semi-automatically labeling objects of interest within an image with corresponding annotations. These annotations provide ground truth information that helps train object detection models to identify and localize objects in new, unseen images.
2. Bounding Boxes: A bounding box is a rectangular region defined by four points: the coordinates of the top-left corner (x, y) and the bottom-right corner (x+w, y+h), where w and h are the width and height of the box, respectively. The bounding box surrounds the object of interest in the image. Each bounding box is associated with a specific class label that identifies the type of object it encloses.
3. Labeling Objects: During the annotation process, an annotator manually draws bounding boxes around the objects in the image using specialized annotation tools or software. The annotator also assigns class labels to each bounding box, indicating the type of object it represents (e.g., car, person, dog, etc.).

![elephant](https://github.com/farastu-who/GIS-DL/assets/34352153/dbcf6eff-3312-44f3-a30f-0e940b4b17f7)

#### Data Generators
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
#### Input Shape & Number of Classes:
The input shape refers to the shape or dimensions of the input data that the model expects. In the context of machine learning and deep learning, input data is represented in the form of a matrix or tensor. The number of classes refers to the total number of distinct categories or labels in a classification task. In classification, the goal is to assign each input data point to one of these classes.

#### Train/Validation/Test Split:
Once the dataset has been pre-processed, we are ready for the final step in setting up the data by creating the Split dataset into 3 sets: “train”, “validation”, and “test” splits (e.g., 60%/20%/20% train/val/test split)
   
### d) Models and Hyperparameters:

A Convolutional Neural Network (CNN) is a type of deep learning model commonly used for image recognition, computer vision tasks, and other pattern recognition problems. CNNs are designed to automatically and adaptively learn spatial hierarchies of features from input images, making them particularly effective in capturing local patterns and structures.

An explanation of a basic CNN architecture model, along with the commonly used hyperparameters, have been delineated below:

#### CNN Architecture Model:

<img width="656" alt="image" src="https://github.com/farastu-who/GIS-DL/assets/34352153/c7dd6e5e-e4e9-4f19-8f17-7bd4777d6236">


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
The Dense layer constructor accepts two main hyper-parameters; units and activation_function. Units are the number of neurons in the layer. In the first Dense layer, there are 128 neurons, and in the second Dense layer, the number of neurons is equal to the number of classes (num_classes) in the dataset.

#### Hyperparameters:

The model learns parameters like the weights and biases, and hyperparameters are parameters that are set before training the model and control various aspects of the learning process. Some common hyperparameters in a CNN include:

1. Number of Convolutional Layers and Filters: The number of convolutional layers and the number of filters in each layer determine the depth and complexity of the CNN architecture. Deeper networks can learn more complex features but may require more computational resources.

2. Kernel Size: The size of the kernels (filters) used in the convolutional layers. Common kernel sizes are 3x3, 5x5, and 7x7.

3. Pooling Size: The size of the pooling windows used in the pooling layers. Common pooling sizes are 2x2 and 3x3.

4. Stride: The step size of the filter as it slides over the input image during convolution. A larger stride reduces the size of the output feature map.

5. Padding: Padding adds extra border pixels to the input image to ensure that the convolution operation does not shrink the spatial dimensions too much. It can be "valid" (no padding) or "same" (pad to retain the spatial dimensions).

6. Activation Function: The choice of activation function should be based on the specific problem, architecture, and dataset. ReLU and its variants are generally preferred due to their simplicity, computational efficiency, and effectiveness in many deep-learning tasks. However, experimenting with different activation functions can sometimes lead to improved performance or faster convergence

7. Number of Fully Connected Layers and Neurons: The number of fully connected layers and the number of neurons in each layer determine the depth and capacity of the fully connected part of the CNN.

8. Learning Rate: The learning rate controls the step size during gradient descent optimization. It determines how much the model's weights are updated during training.

9. Batch Size: The number of samples used in each training iteration. Larger batch sizes can speed up training but may require more memory.

10. Number of Epochs: The number of times the model goes through the entire training dataset during training.

These hyperparameters are crucial for building an effective CNN architecture and are often tuned through experimentation to achieve the best performance on a specific task and dataset. Different combinations of hyperparameters can significantly impact the model's training time, convergence, and generalization ability.

### e) Transfer Learning & Pre-trained Models

A deep learning approach called transfer learning uses a model that has been trained on one problem and uses that training as the foundation for learning how to solve related problems. Pre-trained models created for benchmark datasets like ImageNet can be utilized again in computer vision applications to reduce training time and improve performance.

The procedure entails incorporating one or more layers from a pre-trained model into a new model that has been trained on the particular topic of interest. This may be accomplished in a number of ways, including initializing the weights of the new model based on the pre-trained model or utilizing the pre-trained model as a feature extractor.

Transfer learning is useful because it enables us to take the skills we've developed for one problem—like distinguishing between cats and dogs—and apply them to another—like distinguishing between ants and wasps—even when the target categories are different. We may make use of models' capacity to recognize generic characteristics in images and obtain state-of-the-art performance by utilizing models that have been trained on huge datasets with various categories.

Many of the best-performing models, including as `VGG`, `Inception`, and `ResNet`, which were trained on the `ImageNet` dataset, are readily accessible through Keras and other deep-learning tools.

<img width="768" alt="image" src="https://github.com/farastu-who/GIS-DL/assets/34352153/3aa41ac0-537b-4fe8-94ff-4402f3f9786e">

#### Steps for transfer learning:

1. Obtain Pre-trained Models: The first step is to choose and acquire pre-trained models that were developed using a large-scale image dataset, such ImageNet. These models have learned general features from diverse images and can be a good starting point for a specific problem.

2. Data Preparation: At this stage, we need to collect and label images. Ensure that the images are preprocessed and standardized to fit the input requirements of the pre-trained models. This may involve resizing the images, normalizing pixel values, and augmenting the data to increase its diversity and size.

3. Selecting layers to transfer:  Now, we can choose the pre-trained model's layers that we wish to employ for transfer learning. In most cases, the earlier layers pick up on more general features like edges and textures, whereas the later layers pick up on more detailed aspects. Depending on the size of your dataset and the complexity of your challenge, you may decide whether to employ all or part of these layers.

4. Transfer Learning Strategy: The two main strategies are:
   * Feature Extraction: Use the pre-trained model as a fixed feature extractor by freezing its weights. A new classifier appropriate for the specific task should be added in place of the pre-trained model's initial classification head. Keep the previously trained layers frozen and only train the newly inserted layers.
   * Fine-tuning: This method updates the previously trained layers during the new training process while also replacing the original classifier head. A bigger dataset is necessary for fine-tuning in order to prevent overfitting.

5. Training: Apply the chosen transfer learning strategy to the updated model's training. Observe how the model performs on a validation set, and make any required hyperparameter adjustments. Evaluate the model on a separate test set to ascertain its ability to generalize.

6. Evaluation: Evaluation

7. Iterative Process: Transfer learning frequently involves iterations. The performance of the model may be further enhanced by experimenting with various pre-trained models, layer combinations, or data augmentation strategies, depending on the outcomes.

<img width="1105" alt="image" src="https://github.com/farastu-who/GIS-DL/assets/34352153/54e9b6bd-4d12-4a4f-9b40-3dc140ae8589">


### f) Scoring & Visualization Mechanisms:

In machine learning, particularly when performing classification tasks, recall, precision, accuracy, and F1 score are often used as evaluation metrics. Each metric offers insightful data regarding how well a classification model is doing. The definitions of each metric are as follows:

* Confusion Matrix:
A confusion matrix is a table that provides a summary of the model's predictions and the actual outcomes on a test dataset. The matrix compares the predicted class labels against the true class labels, and it consists of four values:

1. True Positive (TP): The number of instances correctly predicted as the positive class.
2. True Negative (TN): The number of instances correctly predicted as the negative class.
3. False Positive (FP): The number of instances incorrectly predicted as the positive class.
4. False Negative (FN): The number of instances incorrectly predicted as the negative class.

The confusion matrix is particularly useful for understanding the model's performance, especially in cases of imbalanced datasets, where one class may dominate the others. From the confusion matrix, several metrics can be derived, such as accuracy, precision, recall (sensitivity), specificity, and F1-score, which provide insights into the model's strengths and weaknesses.

![image](https://github.com/farastu-who/GIS-DL/assets/34352153/b67d0a45-aa2d-429d-9725-315330e9ded2)


1. Recall (Sensitivity or True Positive Rate):
Recall measures the ability of a model to correctly identify positive instances out of all the actual positive instances. It is calculated as the ratio of true positive (TP) predictions to the sum of true positives and false negatives (FN).
Recall = TP / (TP + FN)

High recall means that the model is good at identifying the positive instances, making it suitable for tasks where false negatives are critical to avoid, such as medical diagnoses, fraud detection, or security applications.

2. Precision (Positive Predictive Value):
Precision measures the accuracy of positive predictions made by the model, i.e., the percentage of correctly predicted positive instances out of all the predicted positive instances. It is calculated as the ratio of true positive (TP) predictions to the sum of true positives and false positives (FP).
Precision = TP / (TP + FP)

High precision means that when the model predicts a positive instance, it is likely to be correct, making it suitable for tasks where false positives are undesirable, such as email spam classification or product defect detection.

3. Accuracy:
Accuracy measures the overall correctness of the model's predictions, irrespective of the class. It is calculated as the ratio of the number of correct predictions (true positives and true negatives) to the total number of instances.
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Accuracy provides a general sense of how well the model performs across all classes. However, it can be misleading in cases of imbalanced datasets, where one class heavily outweighs others.

4. F1 Score:
F1 score is the harmonic mean of precision and recall and provides a balanced measure between the two. It combines both metrics to give a single score that considers false positives and false negatives. The F1 score ranges from 0 to 1, where 1 represents perfect precision and recall.
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

The F1 score is especially useful when dealing with imbalanced datasets or when both precision and recall are equally important in the application.

* Training vs. Validation Loss Graph:
During the training phase, a model learns to make predictions by adjusting its parameters (weights and biases) based on the training data. The training loss represents the error or the difference between the predicted output and the true target labels during training. As the model iteratively updates its parameters, the training loss should decrease over time.
On the other hand, the validation loss measures the model's performance on a separate validation dataset that the model has not seen during training. The validation loss helps to monitor how well the model generalizes to new, unseen data. It is essential to monitor the validation loss to avoid overfitting, which occurs when the model performs well on the training data but poorly on new data.

The training vs. validation loss graph plots the training loss and the validation loss as the model trains over multiple epochs (iterations). We can visualize the learning process on the graph and tell if the model is overfitting or underfitting. In a well-generalized model, the training loss and validation loss should decrease and converge together. If the validation loss starts to increase while the training loss continues to decrease, it indicates overfitting.

* Training vs. Validation Accuracy Graph:
Similar to the training vs. validation loss graph, the training vs. validation accuracy graph plots the training accuracy and validation accuracy over the training process. The training accuracy represents the percentage of correctly predicted instances on the training data, while the validation accuracy measures the performance of the validation dataset.

As the model trains, both the training accuracy and validation accuracy should increase, indicating that the model is learning to make accurate predictions. In an ideal scenario, the training accuracy and validation accuracy would converge, indicating good generalization.

However, if the training accuracy continues to increase while the validation accuracy plateaus or starts to decrease, it suggests overfitting. This means that the model has become too specialized in the training data and struggles to perform well on new data.

![image](https://github.com/farastu-who/GIS-DL/assets/34352153/a0b2c391-8e78-457a-b8bb-12588559f18b)


### g) Inference Integration:
The process of integrating a deep learning or machine learning model that has been trained to make predictions into a system or application is known as inference integration. In order to produce usable outputs from input data, it entails deploying the trained model for usage in the real world.

The following steps are included in the inference integration process:

1. Model Deployment: This step involves migrating the trained and tested model from a development environment (such as a Python script) to a format suitable for deployment in a production environment. 

2. Input Data Preprocessing: The model's input data has to be properly prepared and preprocessed. To make sure the data is in the format the model expects, this may require normalizing, scaling, or encoding.

3. Model Execution: The deployed model is executed on the new, unseen input data. Predictions or inferences are generated by the model using the preprocessed data as input. For instance, the model may be able to predict the class label of a picture in image classification.

4. Post-processing: Depending on the needs of the application, the model's output may need to be further processed or interpreted. This can entail translating model predictions into understandable representations for humans or performing certain actions in response to the model's outputs.

5. Integration with Application: The results of the model's inference are integrated into the larger application or system where the predictions are needed. This could be part of a web application, a mobile app, an automation system, or any other application that can benefit from the model's predictions.

### h) Challenges:

1. Overfitting: 
Overfitting is a common issue in machine learning where a model learns the training data too well and becomes overly specialized to the specific examples in the training set. As a result, the overfitted model performs very well on the training data but fails to generalize to new, unseen data from the real-world, leading to poor performance on the test or validation data. When a model overfits, it memorizes the noise and random variations in the training data rather than learning the underlying patterns that would enable it to make accurate predictions on new data. The model becomes too complex, with too many parameters or features, allowing it to fit even the smallest details in the training data.

2. Underfitting: 
Underfitting occurs when a model is too simple to capture the underlying patterns in the data, leading to poor performance on both the training and test datasets. The model fails to generalize well, and its predictions are inaccurate.

3. Hyperparameter Tuning: 
Machine learning models have hyperparameters that need to be set before training. Poorly tuned hyperparameters can result in suboptimal model performance.

4. Data Quality and Quantity: 
Insufficient or poor data may cause poor model performance. For models to identify significant patterns and generate precise predictions, they need a wide range of representative data.

5. Imbalanced Data: 
Imbalanced datasets occur when one class or category is significantly more prevalent than others. Models trained on imbalanced data may bias their predictions toward the majority class, leading to poor performance on minority classes.

6. Computational Resources: 
Deep learning models, especially large architectures, require significant computational power and memory. Training such models without sufficient resources can lead to slow training or crashes.

![image](https://github.com/farastu-who/GIS-DL/assets/34352153/67ad6a7d-1d40-4937-9d25-f4d1ec8fc267)

#### To address these challenges, several techniques can be employed:

1. Data Augmentation: Increasing the diversity of the training data through data augmentation can help reduce overfitting and improve data quality. For example, in image classification, you can apply random rotations, flips, or translations to generate additional training examples.

2. Regularization: Techniques like L1 or L2 regularization penalize large weights in the model, making it simpler and less prone to overfitting. Dropout is a regularization technique that randomly deactivates some neurons during training, preventing overreliance on specific neurons and enhancing generalization.


3. Cross-Validation: Using techniques like k-fold cross-validation helps to assess the model's performance on different subsets of the data and can give a more reliable estimate of the model's generalization ability.

4. Early Stopping: Monitoring the model's performance on a validation set during training and stopping when performance starts to degrade can prevent overfitting.

5. Representative Data: Ensure that you have enough data to adequately represent the problem domain. If the dataset is small, consider gathering more data or using data augmentation techniques to create additional training examples.

6. Data Preprocessing: Clean and preprocess the data to remove any inconsistencies, outliers, or missing values. Standardize or normalize the features to make them comparable and ensure numerical stability during training.

7. Model Optimization: Optimize the model architecture and hyperparameters to reduce the computational burden. Use smaller model architectures if the task allows, and adjust the hyperparameters to strike a balance between performance and computational resources.

8. Batch Size: Adjust the batch size during training to optimize the trade-off between computation speed and memory usage. Larger batch sizes can be more efficient on GPUs, but they require more memory.

### i) Auto-ML & MLOps

AutoML, short for Automated Machine Learning, aims to democratize machine learning by automating and simplifying the process of building and deploying models. Traditionally, developing machine learning models required expert data scientists with a deep understanding of algorithms, hyperparameters tuning, and feature engineering. However, AutoML enables non-experts to harness the power of machine learning and make data-driven decisions. AutoML is user-friendly and helps accelerate the machine learning development lifecycle by automatically exploring and comparing various models. But some critics argue that it can oversimplify the machine-learning process, leading to potential black-box models with little interpretability. 

Another key aspect when it comes to deploying these large AI models is Machine Learning Operations (MLOps) which bridges the gap between data scientists and IT operations teams, aiming to streamline the deployment and maintenance of machine learning models in production environments. MLOps ensures that machine learning models are not only accurate and performant but also scalable, reliable, and maintainable in real-world scenarios.

#### An example setup of the integration of Auto-ML and MLOps with a machine learning project that uses spatial and satellite data is given below: 
 
1. Using Google Earth Engine (GEE), identify the satellite imagery or geospatial data you want to use for training your neural network models

2. Load the geospatial data (e.g., satellite imagery) into QGIS, and preprocess the data as needed, including tasks such as data clipping, projection conversion, and filtering

3. Export the preprocessed data from QGIS to Google Earth Engine using the Earth Engine Python API and set up Google Colab with required libraries to work on custom neural network models

4. Explore Google Cloud's pre-trained models available in Vertex AI. Choose a pre-trained model that fits your geospatial data analysis needs (e.g., object detection, image classification, etc.).

5. Deploy both your custom-trained model from Colab and the pre-trained model from Vertex AI as endpoints on Vertex AI.

6. Test the deployed models using new data samples and get predictions, then interpret the results and evaluate the effectiveness of the different models. This step should be automated for better efficiency.

7. A version control system like Git should be used to track changes to the machine learning code, data, and model files.

8. Next, the CI (Continuous Integration) pipeline should build the model code, run the automated tests, and provide feedback on the code quality and model performance.

9. The trained model artifacts and other dependencies should be stored in an artifact management system to ensure accessibility.

10. A CD (Continuous Deployment) pipeline should be set up as well to automatically deploy the model to production when all tests pass successfully in the CI pipeline.

11. Use containerization tools like Docker to package the machine learning model and its dependencies into a portable container.

12. Use orchestration tools like Kubernetes to manage and scale the deployment of the containerized machine learning models in a production environment.

13. Finally, implement monitoring and logging for the deployed machine learning models by tracking performance metrics, model accuracy, and other relevant data to detect anomalies and ensure the model's health in production.

<img width="784" alt="image" src="https://github.com/farastu-who/GIS-DL/assets/34352153/63793202-55aa-4b4d-811f-e7765cb3ed1a">


## PART 2: TLC (Transmission Line Classification) - Image Classification of Transmission Lines using Satellite Data

### Project Description:



### Dataset:


#### Further Work: 
-make greyscale
- more data
- labeled data
- models
- object detection
- explore TIFF
- explore several other types of region


#### Resources
1. https://medium.com/spatial-data-science/deep-learning-for-geospatial-data-applications-multi-label-classification-2b0a1838fcf3#:~:text=In%20this%20tutorial%2C%20I%20will%20show%20the%20easiest,lines%20of%20python%20code%20to%20accomplish%20this%20task.
e
2. https://scholar.google.com/citations?user=n1EE3-8AAAAJ
3. 

power line image dataset that are publicly available:
1. TTPLA -
2. Emre, Y.Ö., Nezih, G.Ö., et al.: Power line image dataset (infrared-IR and visible
light-VL). Mendeley Data (2017)
3. https://iopscience.iop.org/article/10.1088/1742-6596/1757/1/012056/pdf
4. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9460718/
5. https://ieeexplore.ieee.org/abstract/document/8550771
6. 
