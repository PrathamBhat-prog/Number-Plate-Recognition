# Number-Plate-Recognition

This is my project where I built a convolutional neural network (CNN) to recognize characters on number plates. I've included the steps I took for data loading, preprocessing, building and training the model, tuning its hyperparameters, and making predictions.

**Data**

I used a dataset for number plate characters. The dataset is organized into training and validation sets, with images categorized by character. I stored my data in Google Drive and accessed it from my Colab notebook.

The images were preprocessed by resizing them to 28x28 pixels and normalizing the pixel values to be between 0 and 1.

**Model Architecture**

Initially, I started with a CNN architecture including several convolutional layers with large kernel sizes, followed by max pooling, flattening, and dense layers.

However, I later changed the architecture and performed hyperparameter tuning to find a better model. The refined architecture that performed best includes:

Two Convolutional layers with 3x3 filters and ReLU activation, followed by MaxPooling layers.
A Flatten layer.
Two Dense layers with ReLU activation, and a final Dense layer with softmax activation for classification into 36 classes (0-9 and A-Z).
Dropout was included after one of the Dense layers for regularization.


**Hyperparameter Tuning**

I used KerasTuner's RandomSearch to find the best hyperparameters for the refined CNN architecture. The tuning objective was to maximize the validation sparse categorical accuracy (val_sparse_categorical_accuracy).

The hyperparameters tuned included:

Number of filters in the first convolutional layer (conv_1_filters): searched values between 16 and 64.
Number of filters in the second convolutional layer (conv_2_filters): searched values between 32 and 128.
Number of units in the first dense layer (dense_1_units): searched values between 128 and 512.
Dropout rate (dropout_1): searched values between 0.2 and 0.5.
Learning rate for the Adam optimizer (learning_rate): searched values between 1e-5 and 1e-3 (log scale).
Results
After hyperparameter tuning, the best model achieved a validation accuracy of approximately 0.9815 and a validation loss of around 0.0882.

**How to Run the Project**
The main code is in the Colab notebook. You can open and run it directly in Google Colab:

Go to Google Colab.
Click on "File" > "Upload notebook" and select the .ipynb file from this repository, OR if you cloned the repository, you can open it from GitHub by clicking "File" > "Open notebook" and navigating to this repository.
Run the cells in the notebook sequentially.
The notebook covers:

Loading and preprocessing the dataset.
Building and training the initial CNN model.
Changing the architecture and performing hyperparameter tuning with KerasTuner.
Saving and loading the trained model.
Making predictions on sample validation images.
Saving and Loading the Model
I have included code in the notebook to save the trained model to an .h5 file and load it back.

**Making Predictions**
The notebook also contains code demonstrating how to load the saved model and use it to make predictions on a new image (or a random image from the validation set).
