

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:06:40 2024

@author: sderodahusman
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
import datetime
# Suppress the specific warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
from matplotlib import rc
import seaborn as sns
# Figure settings
sns.set_style("ticks")
rc('font', weight='bold')


# ---------------------------------------------------------------------
# Step 0: General info
# ---------------------------------------------------------------------

# Define the format of the input string
date_format = "%Y%m%dT%H%M%S"

# ---------------------------------------------------------------------
# Step 1: Select folder
# ---------------------------------------------------------------------

S1_folder = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1clips/AISwide/2021/Draining/'
S1_files = sorted(os.listdir(S1_folder))
S1_files = [file for file in S1_files if file.endswith('.tif')]

S2_folder = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/02_Optical/Data/Predictions_AISwide/Optical_20240318/'

# List of files from the first folder
S2_files = sorted([file for file in os.listdir(S2_folder) if file.endswith('.tif')])



output_folder = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/00_Figures/Draining_AISwide/'

# ---------------------------------------------------------------------
# Step 2: Read files
# ---------------------------------------------------------------------

for i in range(len(S1_files)):
# for i in range(50):

    S1_filename = S1_files[i]
    
    # Get filename information
    S1_filename_we = os.path.splitext(S1_filename)[0]
    S1_filename_split = S1_filename_we.split("_") 
    longitude = S1_filename_split[5]
    latitude = S1_filename_split[4]
    orbit = S1_filename_split[3]
    datasource = S1_filename_split[0]
    
    # Read image
    array_list = []
    ds = gdal.Open(S1_folder + S1_filename)
    array_list.append(ds.ReadAsArray())
    stacked_array = np.stack(array_list, axis=0)[0]
    
    # width_ratios = [2] * int(stacked_array.shape[0])
    height_ratios = [1.5, 1, 1, 1, 1]
    
    fig, axes = plt.subplots(5, 16, figsize=(20,8), dpi=400, gridspec_kw={'height_ratios': height_ratios})
    np.vectorize(lambda axes:axes.axis('off'))(axes) 
    
    bands = {ds.GetRasterBand(i).GetDescription(): i for i in range(1, ds.RasterCount + 1)}
    bandnames = list(bands.keys())
    
    overpasstime = []
    for band_n in range(len(bandnames)):
        date_string = bandnames[band_n].split("_")[4]   
        # Parse the string into a datetime object
        date_object = datetime.datetime.strptime(date_string, date_format).strftime('%Y-%m-%d')
        overpasstime.append(date_object)
    
    
        S1_img = stacked_array[band_n, :, :]
        S1_img = S1_img.astype(float)
        S1_img[S1_img == 0] = np.nan
        S1_img = S1_img / 1e8
        S1_img = 10*np.log10(S1_img)
    

        im = axes[0,band_n].imshow(S1_img, cmap='gray')
        
        axes[0,band_n].set_title(date_object, fontsize='12', weight='bold')
            
    
        if band_n == 0:
            scalebar = ScaleBar(10, location='lower right', units='m', scale_loc='bottom', length_fraction=0.6)   
            axes[0,0].add_artist(scalebar)
        
        ds = None  # Close the datase
        
    
    # fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.95, 0.73, 0.005, 0.12])
    cbar = fig.colorbar(im, cax=cbar_ax,cmap='gray')
    cbar.set_label('Backscatter [dB]', fontsize='10', weight='bold')
    
    
    ######
    S1_latitude ="{:.3f}".format(float(latitude))
    S1_longitude = "{:.3f}".format(float(longitude))
    
    # for i in range(len(S2_files)):
        
        
    count_S2 = 0
    for i in range(len(S2_files)):
        
        S2_filename = S2_files[i]
        
        # Get filename information
        S2_filename_we = os.path.splitext(S2_filename)[0]
        S2_filename_split = S2_filename_we.split("_") 
        S2_longitude = S2_filename_split[6]
        S2_latitude = S2_filename_split[5]
        optical_satellite = S2_filename_split[8]
        
        S2_latitude ="{:.3f}".format(float(S2_latitude))
        S2_longitude = "{:.3f}".format(float(S2_longitude))
        
        S2_overpasstime = S2_filename_split[7]
        S2_date_time = datetime.datetime.fromtimestamp(int(S2_overpasstime)/1000.0)
        S2_date = S2_date_time.strftime('%Y-%m-%d')
        
        # print(S2_latitude)
        # print(S2_longitude)
        if S2_latitude == S1_latitude:
            if S2_longitude == S1_longitude:
                print(S1_longitude, S2_longitude, '.....', S1_latitude, S2_latitude)

                # Read image
                S2_array_list = []
                # Determine which folder the file belongs to
                S2_ds = gdal.Open(S2_folder + S2_filename)

        
                S2_array_list.append(S2_ds.ReadAsArray())
                S2_ds = None  # Close the dataset
                S2_stacked_array = np.stack(S2_array_list, axis=0)[0]
                
                # # Rotate L8 images
                if datasource == 'Sommer':
                    if optical_satellite == 'L8':
                        S2_stacked_array = np.rot90(S2_stacked_array, axes=(1, 2))
                
                # Clip arrays
                S2_stacked_array = S2_stacked_array[:, 50:150, 50:150]
                
                # Plot data
                if count_S2 < 16: 
                    axes[1, count_S2].imshow(S2_stacked_array.transpose(1, 2, 0))
                    axes[1, count_S2].set_title(S2_date, fontsize='12', weight='bold')
                    
                if count_S2 > 15 and count_S2 < 32: 
                    axes[2, count_S2-16].imshow(S2_stacked_array.transpose(1, 2, 0))
                    axes[2, count_S2-16].set_title(S2_date, fontsize='12', weight='bold')
                    
                if count_S2 > 31 and count_S2 < 48: 
                    axes[3, count_S2-32].imshow(S2_stacked_array.transpose(1, 2, 0))
                    axes[3, count_S2-32].set_title(S2_date, fontsize='12', weight='bold')
                    
                if count_S2 > 47 and count_S2 < 64: 
                    axes[4, count_S2-48].imshow(S2_stacked_array.transpose(1, 2, 0))
                    axes[4, count_S2-48].set_title(S2_date, fontsize='12', weight='bold')
            
                count_S2 += 1 
                print(count_S2)
    
    plt.suptitle(f'Draining lake ({S1_longitude}, {S1_latitude}) from {datasource}', y=0.9, weight='bold', fontsize=17)
    fig.savefig(output_folder + 'S1_S2_Draining_' + datasource + '_' + S1_latitude + '_' + S1_longitude + '_' + orbit + '.png', bbox_inches='tight')
    
