#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:21:18 2024

@author: sderodahusman
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from osgeo import gdal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling2D, Flatten, Dense
import numpy as np


# Function to pad arrays to a specified length
def pad_array(array, length):
    padded_array = np.zeros((length,) + array.shape[1:], dtype=array.dtype)
    padded_array[:min(length, array.shape[0])] = array[:min(length, array.shape[0])]
    return padded_array

# Iterate through each file and stack arrays
def load_geotiffs(folder, files, timestep_length=16):
    array_list = []
    for filename in files:
        file_path = os.path.join(folder, filename)
        # Read array
        ds = gdal.Open(file_path)
        array = ds.ReadAsArray() / 1e8   

        if array.shape[0] > 10:
            array_list.append(pad_array(array, timestep_length))
            ds = None  # Close the dataset
        
    stacked_array = np.stack(array_list, axis=0)
    return stacked_array

# Assuming you have functions to load Geotiffs and create ConvLSTM model
refreezing_folder = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1Clips/OttoEtAl/LargerArea4km/S1_OttoEtAl_v4/Freezing/'
draining_folder = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1Clips/OttoEtAl/LargerArea4km/S1_OttoEtAl_v4/Draining/'
buried_folder = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1Clips/OttoEtAl/LargerArea4km/S1_OttoEtAl_v4/Buried/'

# List all files in the folder
refreezing_files = [file for file in os.listdir(refreezing_folder) if file.lower().endswith('.tif')]
draining_files = [file for file in os.listdir(draining_folder) if file.lower().endswith('.tif')]
buried_files = [file for file in os.listdir(buried_folder) if file.lower().endswith('.tif')]

# Load Geotiffs
refreezing_data = load_geotiffs(refreezing_folder, refreezing_files)
draining_data = load_geotiffs(draining_folder, draining_files)
buried_data = load_geotiffs(buried_folder, buried_files)

# # Create Labels
# refreezing_labels = np.zeros(len(refreezing_data))
# draining_labels = np.ones(len(draining_data))
# buried_labels = 2*np.ones(len(buried_data))

# # Organize Data and Labels
# data = np.concatenate((refreezing_data, draining_data, buried_data), axis=0)
# labels = np.concatenate((refreezing_labels, draining_labels, buried_labels), axis=0)

# # Shuffle Data
# random_indices = np.random.permutation(len(data))
# data = data[random_indices]
# labels = labels[random_indices]

# # # Split Data
# # train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# # # ##################################################################

# # # Reshape data to match ConvLSTM input shape (timesteps, height, width, channels)
# # train_data = np.expand_dims(train_data, axis=-1)
# # test_data = np.expand_dims(test_data, axis=-1)


# np.save('/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1clips/OttoEtAl/Data_v1.npy', data.astype('float16'))
# np.save('/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1clips/OttoEtAl/Labels_v1.npy', labels.astype('int'))