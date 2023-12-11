#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Read a comma-separated values (csv) file into DataFrame
df_data = pd.read_csv(r"C:/Users/Inca/Documents/Australia/Deakin/2023/T3 2023/datasets/PressureUlcers_AdditionalData_1.csv")
#df_data = pd.read_csv(r"C:/Users/Inca/Documents/Australia/Deakin/2023/T3 2023/datasets/PressureUlcers_AdditionalData_1.csv")


# In[2]:


# Detect missing values. Returns bool values for each element in the dataframe
# 0 means no null values.
df_data.isnull().sum()

# Data type and null value checking
df_data.info()


# In[3]:


df_data


# In[4]:


# copy the original dataset
data_norm = df_data.copy()


# In[5]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Split the data into features (X) and labels (y)
X = data_norm.drop(['Cod_pat', 'Time', 'Label', 'Ch'], axis=1)
y = data_norm['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


# In[6]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load a copy of the original dataset
#df4 = df_data.copy()

# Step 2: Load and Preprocess Data
# Split the data into features (X) and labels (y)
#X = data_norm.drop(['Cod_pat', 'Time', 'Label', 'Ch'], axis=1)
#y = data_norm['Label']

# Step 3: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Define a list of hyperparameters to search over
param_grid = {'C': [0.1, 1, 10],
              'gamma': [0.001, 0.01, 0.1, 1]}

# Step 6: Perform GridSearchCV to tune hyperparameters
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=6)
grid_search.fit(X_train, y_train.values.ravel())

# Step 7: Get the best hyperparameters
best_params = grid_search.best_params_
best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
print(f"Best C:{best_C}")
print(f"Best Gamma:{best_gamma}")

# Step 8: Train the final model with the best hyperparameters
svm_model = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
svm_model.fit(X_train, y_train.values.ravel())

# Step 9: Evaluate model performance on test dataset
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)


# In[7]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Split the dataset into training and testing sets
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


# In[8]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Extract features (X) and labels (y)
X = data_norm[['Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz']]
y = data_norm['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model architecture
num_features = X_train.shape[1]
hidden_units = 4
output_units = len(np.unique(y))

model = models.Sequential()
model.add(layers.InputLayer(input_shape=(num_features,)))

while hidden_units > output_units:
    model.add(layers.Dense(hidden_units, activation='relu'))
    hidden_units -= 5

model.add(layers.Dense(output_units, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Visualize training history
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

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Make predictions on the test set
y_pred = np.argmax(model.predict(X_test), axis=1)

# Create and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')
labels = np.unique(y)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format=".2f")
plt.title('Confusion Matrix')
plt.show()

# Print the position predictions
print("Position Predictions:")
print(y_pred)


# In[9]:


from tensorflow import keras

# Define the model architecture
num_features = X_train.shape[1]
hidden_units = 10
output_units = len(np.unique(y))

model = models.Sequential()
model.add(layers.InputLayer(input_shape=(num_features,)))

while hidden_units > output_units:
    model.add(layers.Dense(hidden_units, activation='relu'))
    hidden_units -= 5

model.add(layers.Dense(output_units, activation='softmax'))

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Visualize training history
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

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Make predictions on the test set
y_pred = np.argmax(model.predict(X_test), axis=1)

# Create and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')
labels = np.unique(y)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format=".2f")
plt.title('Confusion Matrix')
plt.show()

# Make predictions on new data
# Replace 'new_data' with the actual input data for which you want to make predictions
#new_data = pd.DataFrame({'Ax': [-1.33], 'Ay': [1.1], 'Az': [9.65], 'Mx': [43.27], 'My': [1.09], 'Mz': [-57.76], 'Ch': [1.44]})
#new_data = pd.DataFrame({'Ax': [-1.33], 'Ay': [1.1], 'Az': [9.65], 'Mx': [43.27], 'My': [1.09], 'Mz': [-57.76]})
#new_data = scaler.transform(new_data)  # Standardize the new data using the same scaler
#predictions = model.predict(new_data)

# Print the position predictions
print("Position Predictions:")
print(y_pred)


# In[10]:


cm


# In[ ]:




