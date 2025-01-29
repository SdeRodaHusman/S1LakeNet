


import os
import numpy as np
import pandas as pd
from osgeo import gdal
from keras.models import load_model

def normalize(data_i, min_percentile, max_percentile):
        
    # Convert 0's to NaN so it won't get messed up when converting to log-scale
    data_i[data_i == 0] = np.nan
    
    # Convert to log-scale
    data_i = 10*np.log10(data_i)

   
    # Flatten array
    flat = data_i.flatten()
    
    # Calculate the minimum and maximum values of the data
    min_val = np.nanpercentile(flat, min_percentile)
    max_val = np.nanpercentile(flat, max_percentile)

    # Min-max normalize the data to the range [0, 1]
    data_i = (data_i - min_val) / (max_val - min_val)
    
    # Convert back to zeros
    data_i[np.isnan(data_i)] = 0
    
    # data_sample[data_sample < 0] = 0
    return data_i
    
    
data_norm = []

# Function to pad arrays to a specified length
def pad_array(array, length):
    padded_array = np.zeros((length,) + array.shape[1:], dtype=array.dtype)
    padded_array[:min(length, array.shape[0])] = array[:min(length, array.shape[0])]
    return padded_array

# Iterate through each file and stack arrays
def load_geotiffs(folder, files, timestep_length=16):
    array_list = []
    # for filename in files:
    file_path = os.path.join(folder, files)
    # Read array
    ds = gdal.Open(file_path)
    array = ds.ReadAsArray() / 1e8
    
    # Check the original size of the array before padding
    if array.shape[0] >= 5:
        padded_array = pad_array(array, timestep_length)
        array_list.append(padded_array)
    
    ds = None  # Close the dataset
    
    # Check if any arrays are appended to array_list
    if array_list:
        stacked_array = np.stack(array_list, axis=0)
        return stacked_array
    else:
        return None

# Load the model
version = 'v6'
model = load_model('/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/Tests/Models/Convlstm_model_' + version + '.h5')

# Select year of interest
lakeMode = 'Freezing'

# Directory containing the files
folder = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1clips/OttoEtAl/' + lakeMode + '/'

# Initialize lists to store predictions, latitudes, longitudes, and orbits
predictions_list = []
predictions_nr_list = []
latitudes = []
longitudes = []
orbits = []


# Iterate through files in the directory
for i in os.listdir(folder):
    if i.lower().endswith('.tif'):
        # Load geotiff data
        data = load_geotiffs(folder, i)
        
        # If the original size of the array is smaller than 5 or None, skip this file
        if data is None:
            continue
        print(i)
        # Extract latitude, longitude, and orbit from the filename
        parts = os.path.splitext(i)[0].split('_')
        longitude = float(parts[7])
        latitude = float(parts[8])
        orbit = int(parts[9])
#         tile = int(parts[3])
#         area = int(parts[8])
#         volume = int(parts[10])
        
        # Append to the lists
        longitudes.append(longitude)
        latitudes.append(latitude)
        orbits.append(orbit)
#         tiles.append(tile)
#         areas.append(area)
#         volumes.append(volume)
        
        # Normalize data
        data = normalize(data, 2.5, 97.5)
    
        # Make predictions
        predictions = model.predict(data)[0][0]
        predictions_nr_list.append(predictions)
        
        # Round predictions
        # rounded_predictions = np.round(predictions)
        rounded_predictions = np.where(predictions >= 0.6, 1, 0)

    
        predictions_list.append(rounded_predictions)
        
        
# Create a DataFrame
data = {'Prediction_NR': predictions_nr_list, 'Latitude': latitudes, 'Longitude': longitudes, 'Orbits': orbits}
df = pd.DataFrame(data)

# Check for duplicates based on longitude and latitude
duplicates = df[df.duplicated(['Longitude', 'Latitude'], keep=False)]

# Create a new DataFrame with unique longitude and latitude
unique_coords_df = df.drop_duplicates(['Longitude', 'Latitude'])

# Calculate the average predictions for each unique longitude and latitude
average_predictions = df.groupby(['Longitude', 'Latitude'])['Prediction_NR'].mean().reset_index()

# Calculate the number of observations for each unique longitude and latitude
num_observations = df.groupby(['Longitude', 'Latitude']).size().reset_index(name='NumObservations')

# Merge the unique coordinates DataFrame with the average predictions and number of observations
df = pd.merge(unique_coords_df, average_predictions, on=['Longitude', 'Latitude'])
df = pd.merge(df, num_observations, on=['Longitude', 'Latitude'])

# Save DataFrame as CSV
# output_path = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1clips/AISwide/Predictions_LakeEvolution_' + year + '_' + version + '.csv'
# df.to_csv(output_path, index=False)


df['Prediction_R05'] = np.where(df['Prediction_NR_x'] >= 0.5, 1, 0)
df['Prediction_R06'] = np.where(df['Prediction_NR_x'] >= 0.6, 1, 0)
df['Prediction_R07'] = np.where(df['Prediction_NR_x'] >= 0.7, 1, 0)
df['Prediction_R08'] = np.where(df['Prediction_NR_x'] >= 0.8, 1, 0)
df['Prediction_R09'] = np.where(df['Prediction_NR_x'] >= 0.9, 1, 0)

# # count values in marks column
# print('Drainages in Prediction_R05: ', df['Prediction_R05'].value_counts()[1])
print('Drainages in Prediction_R06: ', df['Prediction_R06'].value_counts()[1]/df.shape[0])
# print('Drainages in Prediction_R07: ', df['Prediction_R07'].value_counts()[1])
# print('Drainages in Prediction_R08: ', df['Prediction_R08'].value_counts()[1])
# print('Drainages in Prediction_R09: ', df['Prediction_R09'].value_counts()[1])



# # # Save DataFrame as CSV
# output_path = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1clips/AISwide/Predictions_LakeEvolution_' + year + '_' + version + '.csv'
# df.to_csv(output_path, index=False)
