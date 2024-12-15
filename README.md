# Fire Detection using CNN

This repository implements a Fire Detection System using Convolutional Neural Networks (CNNs). The goal is to classify images into "fire" or "non-fire" categories by leveraging modern deep learning techniques. This system has been designed to assist in early fire detection through image data processing, making it potentially applicable in real-world scenarios such as surveillance systems.

#Fire Detection
Fire detection is a critical task in preventing catastrophic damages caused by uncontrolled fires. Traditional fire detection systems rely heavily on physical sensors like smoke or heat detectors. However, with the advancement of computer vision and deep learning, image-based fire detection provides a complementary approach. 

A CNN, designed to automatically and adaptively learn spatial hierarchies of features, is highly effective in tasks like image classification. By training on labeled images of fire and non-fire, the CNN learns to distinguish between the two, even under varied conditions such as different lighting or background noise. 

The implementation includes data preprocessing, augmentation, model training, validation, and prediction. Advanced techniques like dropout layers reduce overfitting, while checkpoints ensure only the best-performing models are saved.

## Features
- Binary classification of images as "fire" or "non-fire."
- Automated data augmentation to improve model generalization.
- Dropout layers to reduce overfitting.
- Model checkpointing to save the best-performing model during training.
- Support for testing on new unseen images.

## Technologies Used
- Python
- TensorFlow
- Keras
- Google Colab
- NumPy
- ImageDataGenerator (for data augmentation)

## Directory Structure
Fire_detection/
├── train/                 # Training data directory
│   ├── fire/             # Images with fire
│   ├── non-fire/         # Images without fire
├── test/                  # Test data directory
│   ├── test_image.jpg    # Example test image
├── model/                 # Saved models directory
│   ├── model.keras       # Best model saved during training
└── README.md              # Project documentation


## Model Architecture
The CNN model includes the following layers:

- Convolutional Layers: Extract spatial features such as edges and textures.
- Max Pooling Layers: Reduce dimensionality while preserving important features.
- Fully Connected Layers: Transform learned features into final classifications.
- Dropout Layers: Introduced to mitigate overfitting during training.
- Output Layer: Binary classification using a sigmoid activation function.

## Summary of Model Layers
Conv2D: Extracts features from the input image, outputting a feature map of size (256, 256, 32) with 896 parameters.
MaxPooling2D: Reduces the size of the feature map to (128, 128, 32) without learning any parameters.
Dense: Combines features for final classification, outputting a single value with 129 parameters.

## Training the Model
### Key Steps
1. Data Augmentation
   To increase the diversity of training data

2. Model Checkpoints
   Save the best model based on validation loss

3. Model Training
   Train the model with the augmented dataset
   
### Example Training Output
Epoch 1/2: The model achieves an accuracy of 80.05% on the training set, with a loss of 3.5406. On the validation set, it reaches 84.38% accuracy, indicating good performance but some room for improvement.

Epoch 2/2: The accuracy improves to 87.50% on the training set, with a lower loss of 0.3977. The validation accuracy reaches 100%, with a very low loss of 0.0322, indicating strong generalization to new data.

## Testing the Model
Testing the Model
In this section, we evaluate the trained CNN model on unseen images to predict whether they contain fire or not.

Loading and Preprocessing: First, an image is loaded and resized to match the input size expected by the model (256x256 pixels). The image is then converted to a numpy array and normalized by dividing by 255 to scale pixel values between 0 and 1.

Batch Dimension: Since the model expects a batch of images (even if it's just one image), we expand the image's dimensions to simulate a batch by using np.expand_dims().

Prediction: The model makes a prediction using the predict() method. The output is a probability value between 0 and 1. A value closer to 1 indicates the presence of fire, while a value closer to 0 indicates no fire.

Binary Classification: To convert the probability into a class label (0 or 1), we apply a threshold of 0.5. If the predicted probability is greater than 0.5, the class is labeled as "fire" (1), otherwise, it’s labeled as "non-fire" (0).
from tensorflow.keras.preprocessing import image
import numpy as np


## Limitations and Future Work
- Data Limitation: Performance may vary on diverse datasets. Adding more data could improve accuracy.
- Real-world Testing: Model needs testing in real surveillance settings to evaluate robustness.
- Hyperparameter Tuning: Optimizing learning rates, batch sizes, and architecture depth could enhance performance.

