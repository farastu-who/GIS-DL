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
5. TTPLA: This is a public dataset that is a collection of aerial images of Transmission Towers (TTs) and Powers Lines (PLs). It consists of 1,100 images with a resolution of 3,840×2,160 pixels, as well as manually labeled 8,987 instances of TTs and PLs. An example from the dataset is shown below.

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

Train/Validation/Test Split:
Once the dataset has been pre-processed, we are ready for the final step in setting up the data by creating the Split dataset into 3 sets: “train”, “validation”, and “test” splits (e.g., 60%/20%/20% train/val/test split)
   
#### d) Models and Hyperparameters:

A Convolutional Neural Network (CNN) is a type of deep learning model commonly used for image recognition, computer vision tasks, and other pattern recognition problems. CNNs are designed to automatically and adaptively learn spatial hierarchies of features from input images, making them particularly effective in capturing local patterns and structures.

An explanation of a basic CNN architecture model, along with the commonly used hyperparameters, have been delineated below:

##### CNN Architecture Model:

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

##### Hyperparameters:

The model learns parameters like the weights and biases, and hyperparameters are parameters that are set before training the model and control various aspects of the learning process. Some common hyperparameters in a CNN include:

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

#### e) Transfer Learning & Pre-trained Models

A deep learning approach called transfer learning uses a model that has been trained on one problem as the foundation for learning how to solve related problems. Pre-trained models created for benchmark datasets like ImageNet can be utilized again in computer vision applications to reduce training time and improve performance.

The procedure entails incorporating one or more layers from a pre-trained model into a new model that has been trained on the particular topic of interest. This may be accomplished in a number of ways, including initializing the weights of the new model based on the pre-trained model or utilizing the pre-trained model as a feature extractor.

Transfer learning is useful because it enables us to take the skills we've developed for one problem—like distinguishing between cats and dogs—and apply them to another—like distinguishing between ants and wasps—even when the target categories are different. We may make use of models' capacity to recognize generic characteristics in images and obtain state-of-the-art performance by utilizing models that have been trained on huge datasets with various categories.

Many of the best-performing models, including as `VGG`, `Inception`, and `ResNet`, which were trained on the `ImageNet` dataset, are readily accessible through Keras and other deep-learning tools.

<img width="768" alt="image" src="https://github.com/farastu-who/GIS-DL/assets/34352153/3aa41ac0-537b-4fe8-94ff-4402f3f9786e">

Steps for transfer learning:

1. Obtain Pre-trained Models: The first step is to choose and acquire pre-trained models that were developed using a large-scale image dataset, such ImageNet. These models have learned general features from diverse images and can be a good starting point for a specific problem.

2. Data Preparation: At this stage, we need to collect and label images. Ensure that the images are preprocessed and standardized to fit the input requirements of the pre-trained models. This may involve resizing the images, normalizing pixel values, and augmenting the data to increase its diversity and size.

3. Selecting layers to transfer:  Now, we can choose the pre-trained model's layers that we wish to employ for transfer learning. In most cases, the earlier layers pick up on more general features like edges and textures, whereas the later layers pick up on more detailed aspects. Depending on the size of your dataset and the complexity of your challenge, you may decide whether to employ all or part of these layers.

4. Transfer Learning Strategy: The two main strategies are:
   * Feature Extraction: Use the pre-trained model as a fixed feature extractor by freezing its weights. A new classifier appropriate for the specific task should be added in place of the pre-trained model's initial classification head. Keep the previously trained layers frozen and only train the newly inserted layers.
   * 
Fine-tuning: Using this method, you may update the previously trained layers during training while also replacing the original classifier head. You "fine-tune" these layers to make them fit your transmission line dataset's particulars. A bigger dataset is necessary for fine-tuning in order to prevent overfitting.

Training and Evaluation: Employ the selected transfer learning approach to train your updated model. Keep track of how the model performs on a validation set and adjust the hyperparameters as necessary. To determine the model's capacity for generalization, evaluate it on a different test set.

Iterative Process: Transfer learning frequently involves iterations. The performance of the model may be further enhanced by experimenting with various pre-trained models, layer combinations, or data augmentation strategies, depending on the outcomes.
<img width="1105" alt="image" src="https://github.com/farastu-who/GIS-DL/assets/34352153/54e9b6bd-4d12-4a4f-9b40-3dc140ae8589">


#### f) Scoring & Visualization Mechanisms:

Confusion Matrix:
A confusion matrix is a table that is often used to evaluate the performance of a machine learning or classification model. It provides a summary of the model's predictions and the actual outcomes on a test dataset. The matrix compares the predicted class labels against the true class labels, and it consists of four values:
True Positive (TP): The number of instances correctly predicted as the positive class.
True Negative (TN): The number of instances correctly predicted as the negative class.
False Positive (FP): The number of instances incorrectly predicted as the positive class (Type I error).
False Negative (FN): The number of instances incorrectly predicted as the negative class (Type II error).
The confusion matrix is particularly useful for understanding the model's performance, especially in cases of imbalanced datasets, where one class may dominate the others. From the confusion matrix, several metrics can be derived, such as accuracy, precision, recall (sensitivity), specificity, and F1-score, which provide insights into the model's strengths and weaknesses.

Training vs Validation Loss Graph:
In machine learning, during the training phase, a model learns to make predictions by adjusting its parameters (weights and biases) based on the training data. The training loss represents the error or the difference between the predicted output and the true target labels during training. As the model iteratively updates its parameters, the training loss should decrease over time.
On the other hand, the validation loss measures the model's performance on a separate validation dataset that the model has not seen during training. The validation loss helps to monitor how well the model generalizes to new, unseen data. It is essential to monitor the validation loss to avoid overfitting, which occurs when the model performs well on the training data but poorly on new data.

The training vs validation loss graph plots the training loss and the validation loss as the model trains over multiple epochs (iterations). The graph allows us to visualize the learning process and determine if the model is overfitting or underfitting. In a well-generalized model, the training loss and validation loss should decrease and converge together. If the validation loss starts to increase while the training loss continues to decrease, it indicates overfitting.

Training vs Validation Accuracy Graph:
Similar to the training vs validation loss graph, the training vs validation accuracy graph plots the training accuracy and validation accuracy over the training process. The training accuracy represents the percentage of correctly predicted instances on the training data, while the validation accuracy measures the performance on the validation dataset.
As the model trains, both the training accuracy and validation accuracy should increase, indicating that the model is learning to make accurate predictions. In an ideal scenario, the training accuracy and validation accuracy would converge, indicating good generalization.

However, if the training accuracy continues to increase while the validation accuracy plateaus or starts to decrease, it suggests overfitting. This means that the model has become too specialized in the training data and struggles to perform well on new data.

The training vs validation accuracy graph helps to identify the model's performance trends during training and assess whether it is achieving good generalization or suffering from overfitting. Monitoring both loss and accuracy graphs is essential for selecting the best model and fine-tuning hyperparameters to achieve the desired level of performance.

Recall, precision, accuracy, and F1 score are commonly used evaluation metrics in machine learning, particularly in classification tasks. Each metric provides valuable insights into the performance of a classification model. Here's an explanation of each metric:

Recall (Sensitivity or True Positive Rate):
Recall measures the ability of a model to correctly identify positive instances out of all the actual positive instances. It is calculated as the ratio of true positive (TP) predictions to the sum of true positives and false negatives (FN).
Recall = TP / (TP + FN)

High recall means that the model is good at identifying the positive instances, making it suitable for tasks where false negatives are critical to avoid, such as medical diagnoses, fraud detection, or security applications.

Precision (Positive Predictive Value):
Precision measures the accuracy of positive predictions made by the model, i.e., the percentage of correctly predicted positive instances out of all the predicted positive instances. It is calculated as the ratio of true positive (TP) predictions to the sum of true positives and false positives (FP).
Precision = TP / (TP + FP)

High precision means that when the model predicts a positive instance, it is likely to be correct, making it suitable for tasks where false positives are undesirable, such as email spam classification or product defect detection.

Accuracy:
Accuracy measures the overall correctness of the model's predictions, irrespective of the class. It is calculated as the ratio of the number of correct predictions (true positives and true negatives) to the total number of instances.
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Accuracy provides a general sense of how well the model performs across all classes. However, it can be misleading in cases of imbalanced datasets, where one class heavily outweighs others.

F1 Score:
F1 score is the harmonic mean of precision and recall and provides a balanced measure between the two. It combines both metrics to give a single score that considers false positives and false negatives. The F1 score ranges from 0 to 1, where 1 represents perfect precision and recall.
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

The F1 score is especially useful when dealing with imbalanced datasets or when both precision and recall are equally important in the application.

In summary, recall, precision, accuracy, and F1 score are crucial evaluation metrics in classification tasks, and the choice of the appropriate metric depends on the specific requirements and objectives of the machine learning application. Evaluating a model using multiple metrics can provide a comprehensive understanding of its performance and guide further improvements in the model.



#### g) Auto-ML

 
Abstract: Use ML and/or DL models on the training set of annotated and labeled data of the missing transmission lines (above 230kV) as compared to HIFLD and test it on the other transmission lines of lower V.

new from the HIFLD data - compare to HE TL layer - get unmatched -  find raster images for locations - analysis 

#### h) Inference Integration:
Inference integration refers to the process of incorporating a trained machine learning or deep learning model into an application or system to make predictions or inferences on new, unseen data. In other words, it involves deploying the trained model for real-world use to obtain useful outputs from input data.

During the training phase of a machine learning model, the model learns to generalize patterns and relationships from a labeled dataset. Once the model is trained, it can be used for inference on new, unlabeled data. This is when inference integration becomes crucial. Inference integration enables the model to be applied to real-world scenarios, making predictions or classifications based on input data it has not seen during training.

The process of inference integration involves the following steps:

Model Deployment: The trained model is prepared and configured for deployment. This typically involves converting the model from a development environment (e.g., a Python script) to a format suitable for deployment in a production environment. The model might be saved as a binary file or serialized to a format that allows it to be easily loaded and used by other applications or systems.

Input Data Preprocessing: The input data to the model needs to be preprocessed and formatted appropriately. This might involve data normalization, scaling, or encoding to ensure it matches the format that the model expects.

Model Execution: The deployed model is executed on the new, unseen input data. The model takes the preprocessed data as input and produces predictions or inferences as output. For example, in image classification, the model might predict the class label of an image.

Post-processing: The output from the model might need further processing or interpretation, depending on the application's requirements. This could involve converting model predictions into human-readable formats or taking specific actions based on the model's outputs.

Integration with Application: The results of the model's inference are integrated into the larger application or system where the predictions are needed. This could be part of a web application, a mobile app, an automation system, or any other application that can benefit from the model's predictions.

Inference integration is a critical step in the machine learning workflow, as it enables the practical use of machine learning models to make informed decisions and take actions based on new data. The integration process ensures that the model's predictive capabilities are harnessed effectively and seamlessly in real-world applications.





#### i Challenges:

1. Overfitting
2. Underfitting
3. Hyperparameter Tuning
4. Data Quality and Quantity
5. Imbalanced Data
6. Computational Resources

Overfitting is a common issue in machine learning where a model learns the training data too well and becomes overly specialized to the specific examples in the training set. As a result, the overfitted model performs very well on the training data but fails to generalize to new, unseen data from the real-world, leading to poor performance on the test or validation data.

When a model overfits, it memorizes the noise and random variations in the training data rather than learning the underlying patterns that would enable it to make accurate predictions on new data. The model becomes too complex, with too many parameters or features, allowing it to fit even the smallest details in the training data.

The main characteristics of an overfit model are:

High Training Accuracy: An overfitted model achieves very high accuracy on the training data because it has essentially memorized the training examples.

Low Test Accuracy: Despite its high training accuracy, an overfitted model performs poorly on new, unseen data, resulting in low accuracy on the test or validation set.

The consequences of overfitting are problematic as it leads to a lack of generalization. In real-world applications, the primary goal of machine learning is to build models that can perform well on unseen data to make accurate predictions. Overfitting hinders this objective as the model becomes too tailored to the training data, failing to recognize broader patterns that are necessary for generalization.

To address overfitting, several techniques can be employed:

Data Augmentation: Increasing the diversity of the training data through data augmentation can help reduce overfitting. For example, in image classification, you can apply random rotations, flips, or translations to generate additional training examples.

Regularization: Techniques like L1 or L2 regularization penalize large weights in the model, making it simpler and less prone to overfitting.

Cross-Validation: Using techniques like k-fold cross-validation helps to assess the model's performance on different subsets of the data and can give a more reliable estimate of the model's generalization ability.

Early Stopping: Monitoring the model's performance on a validation set during training and stopping when performance starts to degrade can prevent overfitting.

Model Selection: Choosing a simpler model architecture with fewer layers or nodes can reduce overfitting.

Dropout: Dropout is a regularization technique that randomly deactivates some neurons during training, preventing overreliance on specific neurons and enhancing generalization.

By using these strategies, you can mitigate the risk of overfitting and build models that perform well on new, unseen data, leading to more accurate and reliable predictions in real-world applications.






Data Format:
Tab, shp, raster, GeoJSON




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
