#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Wrangling Libraries
import pandas as pd
import numpy as np


# ### About the data

# In[2]:


# Read a comma-separated values (csv) file into DataFrame
df_data = pd.read_csv(r"C:/Users/Inca/Documents/Australia/Deakin/2023/T3 2023/datasets/PressureUlcers_AdditionalData_1.csv")

# Detect missing values. Returns bool values for each element in the dataframe
# 0 means no null values.
df_data.isnull().sum()

# Data type and null value checking
df_data.info()

# Load a copy of the original dataset
data_norm = df_data.copy()


# In[3]:


df_data


# ### SVM and Classification Report

# In[4]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Step 1: Load and Preprocess Data.
# Split the data into features (X) and labels (y).
X = data_norm.drop(['Cod_pat', 'Time', 'Label', 'Ch'], axis=1)
y = data_norm['Label']

# Step 2: Split the data into training and testing sets (80% training and 20% testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the SVM model.
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Step 4: Make predictions on the test set.
y_pred = svm_model.predict(X_test)

# Step 5: Evaluate the model.
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Step 6: Print the results.
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


# In[5]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load and Preprocess Data.
# Split the data into features (X) and labels (y).
X = data_norm.drop(['Cod_pat', 'Time', 'Label', 'Ch'], axis=1)
y = data_norm['Label']

# Step 2: Split the data into training and testing sets (80% training, 20% testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Define a list of hyperparameters to search over.
param_grid = {'C': [0.1, 1, 10],
              'gamma': [0.001, 0.01, 0.1, 1]}

# Step 5: Perform GridSearchCV to tune hyperparameters.
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=6)
grid_search.fit(X_train, y_train.values.ravel())

# Step 6: Get the best hyperparameters.
best_params = grid_search.best_params_
best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
print(f"Best C:{best_C}")
print(f"Best Gamma:{best_gamma}")

# Step 7: Train the final model with the best hyperparameters.
svm_model = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
svm_model.fit(X_train, y_train.values.ravel())

# Step 8: Evaluate model performance on test dataset.
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Step 9: Print the results. 
print("Test Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)


# ### KNN model

# In[6]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Step 1: Load and Preprocess Data.
# Split the data into features (X) and labels (y).
X = data_norm.drop(['Cod_pat', 'Time', 'Label', 'Ch'], axis=1)
y = data_norm['Label']

# Step 2: Split the data into training and testing sets (80% training, 20% testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Split the dataset into training and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose the number of neighbors (k)
k_values = range(1, 10)
best_k = None
best_score = 0

# Iterate over different k values to find the best one
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train.values.ravel(), cv=6)  # 6-fold cross-validation
    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score = mean_score
        best_k = k
        
# Train the KNN model with the best k value
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train.values.ravel())

# Evaluate the KNN model on the test set
knn_accuracy = knn.score(X_test, y_test)
print(f"KNN Accuracy (k={best_k}): {knn_accuracy:.4f}")


# ### TensorFlow neural network

# In[7]:


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import ConfusionMatrixDisplay

# Step 1: Load and Preprocess Data.
# Split the data into features (X) and labels (y).
X = data_norm[['Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz']]
y = data_norm['Label']

# Step 2: Split the data into training and testing sets (80% training and 20% testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Define the model architecture
num_features = X_train.shape[1]

# This line sets the number of hidden units or neurons in the hidden layer 
# of the neural network. In this case, it is set to 4.
hidden_units = 4

# This line calculates the number of output units in the output layer of 
# the neural network.It assumes that y is the target variable, and np.unique(y) 
# returns the unique classes in the target variable.The length of this unique 
# set represents the number of output classes.
output_units = len(np.unique(y))

# This line creates a Sequential model using Keras.
# The Sequential model is a linear stack of layers, where I can simply add one layer at a time.
model = models.Sequential()

# This line adds an input layer to the model.
model.add(layers.InputLayer(input_shape=(num_features,)))

# This line starts a while loop that continues executing the code within
# its block as long as the value of hidden_units is greater than the 
# value of output_units.
while hidden_units > output_units:
    # Inside the loop, this line adds a dense (fully connected) layer to a neural network model. 
    # The Dense layer is a standard layer in neural networks where each neuron in the layer 
    # is connected to every neuron in the previous layer.
    model.add(layers.Dense(hidden_units, activation='relu'))
    
    # After adding a layer, this line decrements the value of hidden_units by 5. 
    # This means that each iteration of the loop reduces the number of neurons in the hidden layer by 5.
    hidden_units -= 5

model.add(layers.Dense(output_units, activation='softmax'))

# Step 5: Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 7: Visualize training history
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Step 8: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Step 8: Make predictions on the test set
y_pred = np.argmax(model.predict(X_test), axis=1)

# Step 9: Create and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')
labels = np.unique(y)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format=".2f")
plt.title('Confusion Matrix')
plt.show()

# Step 10: Print the position predictions
print("Position Predictions:")
print(y_pred)

