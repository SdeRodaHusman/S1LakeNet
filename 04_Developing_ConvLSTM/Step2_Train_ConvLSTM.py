#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:43:29 2024

@author: sderodahusman
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from osgeo import gdal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
# Import TensorBoard callback
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# # Define TensorBoard callback
# log_dir = "/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/04_ModelOutput/TensorBoard/"
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# ---------------------------------------------------------------------
# Step 0: Load data
# ---------------------------------------------------------------------

# Example: Simulate loading two layers per timestep
data_hh = np.load('/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1clips/OttoEtAl/Data_HH_temp.npy')
data_hv = np.load('/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1clips/OttoEtAl/Data_HV_temp.npy')

labels = np.load('/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1clips/OttoEtAl/Labels_temp.npy')

# Combine layers into a single dataset with shape (samples, timesteps, height, width, 2)
data = np.stack((data_hh, data_hv), axis=-1)  # Adds a channel dimension
data[np.isnan(data)] = 0

data = data * 10 

# test = np.nanmean(data, axis=0)
fig_test = data[0,1,:,:,0]
plt.imshow(fig_test)
plt.colorbar()

# data_norm = []

# for n in range(len(data)):
#     data_sample = data[n, :, :, :, :]  # Shape: (timesteps, height, width, channels)
    
#     for c in range(data_sample.shape[-1]):  # Normalize each channel independently
#         flat = data_sample[..., c].flatten()
#         min_val = np.nanpercentile(flat, 2.5)
#         max_val = np.nanpercentile(flat, 97.5)
#         data_sample[..., c] = (data_sample[..., c] - min_val) / (max_val - min_val)
#         data_sample[data_sample[..., c] < 0, c] = 0
    
#     data_norm.append(data_sample)

# data = np.stack(data_norm, axis=0)
# data[np.isnan(data)] = 0

# print(np.min(data), np.max(data))

# ---------------------------------------------------------------------
# Step 1: Make training/testing split
# ---------------------------------------------------------------------

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------
# Step 2: Compute class-weight
# ---------------------------------------------------------------------

class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(train_labels),
                                        y = train_labels                                                    
                                    )
class_weights = dict(zip(np.unique(train_labels), class_weights))

def weighted_binary_crossentropy(y_true, y_pred):
    # Cast y_true to float32 to match the data type of y_pred
    y_true = tf.cast(y_true, tf.float32)
    
    # Clip predictions to prevent log(0) errors
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    # Calculate the weighted binary crossentropy
    loss = -(y_true * tf.math.log(y_pred) * class_weights[1] + (1 - y_true) * tf.math.log(1 - y_pred) * class_weights[0])
    
    return tf.reduce_mean(loss, axis=-1)


# ---------------------------------------------------------------------
# Step 3: Develop and compile ConvLSTM model
# --------------------------------------------------------------------

# # Define the ConvLSTM model
# model = Sequential()
# model.add(ConvLSTM2D(
#     filters=64, 
#     kernel_size=(3, 3), 
#     activation='relu', 
#     input_shape=(16, 100, 100, 2)  # Note the 2 channels here
# ))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', weighted_metrics=['accuracy'])
# # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model with early stopping to prevent overfitting
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model = Sequential()

# ConvLSTM2D layer with L2 regularization
model.add(ConvLSTM2D(
    filters=64, 
    kernel_size=(3, 3), 
    activation='relu', 
    input_shape=(16, 100, 100, 2),
    kernel_regularizer=regularizers.l2(0.001)  # L2 regularization
))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add Dropout to prevent overfitting
model.add(Dropout(0.3))

model.add(Flatten())

# Dense layer with L2 regularization and Dropout
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

# Compile model with lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# ---------------------------------------------------------------------
# Step 4: Train model
# ---------------------------------------------------------------------

# Train the model
model.fit(train_data, train_labels, epochs=15, batch_size=32, 
          validation_data=(test_data, test_labels),
          callbacks=[early_stopping])

# # ---------------------------------------------------------------------
# # Step 5: Save model
# # ---------------------------------------------------------------------

# # Save the trained model
# model.save('/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/Tests/Models/Convlstm_model_v6data_v2model.h5')

# ---------------------------------------------------------------------
# Step 6: Evaluate model
# ---------------------------------------------------------------------

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


