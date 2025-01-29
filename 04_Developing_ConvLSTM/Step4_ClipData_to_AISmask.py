#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:20:38 2024

@author: sderodahusman
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import pandas as pd
import os    
from osgeo import gdal                                                               
import glob  
import seaborn as sns
import numpy as np
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
import datetime
import rasterio
from rasterio.mask import mask
import geopandas as gpd

# ---------------------------------------------------------------------
# Settings:
# ---------------------------------------------------------------------

selectedYear = '2020'   # Options: 2018, 2019, 2020
lakeMode = 'Freeze'

# # ---------------------------------------------------------------------
# # Import shapefile with lakes
# # ---------------------------------------------------------------------

# filename_lakes = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/01_Labels/OttoEtAl/DrainageMode' + selectedYear+ '.shp'
# lakes = gpd.read_file(filename_lakes)

    

# ---------------------------------------------------------------------
# Clip Sentinel-1 data to individual lakes
# ---------------------------------------------------------------------




lakes_basefolder = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/01_Labels/OttoEtAl/IndividualLakeMasks/' + selectedYear + '/'

# Folder with Sentinel-1 data per lake boundary
S1_basefolder = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1Clips/OttoEtAl/01_RawData/PeriodJuneOct/' + lakeMode + '/'

# New base-folder with Sentinel-1 data exactly clipped per lake
S1_clippedfolder = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1Clips/OttoEtAl/02_Clipped/' 
  
for lake_file in os.listdir(lakes_basefolder):
    lakeModeCheck = lake_file.split("_")[0]
    selectedYearCheck = lake_file.split("_")[1]
    if lakeModeCheck == lakeMode:
        if selectedYearCheck == selectedYear:
            centroid_3413 = gpd.read_file(lakes_basefolder + lake_file).geometry.centroid
            # Set the current CRS to EPSG:3413
            centroid_3413 = centroid_3413.set_crs("EPSG:3413")    
            # Reproject the centroid to EPSG:4326
            lakecentroid = centroid_3413.to_crs("EPSG:4326")
            lake_lon = round(float(lakecentroid.y.values[0]), 2)
            lake_lat = round(float(lakecentroid.x.values[0]), 2)
            
            for S1_file in os.listdir(S1_basefolder):
                S1_lakeModeCheck = S1_file.split("_")[2]
                S1_selectedYearCheck = S1_file.split("_")[3]
                S1_lon = round(float(S1_file.split("_")[6]), 2)
                S1_lat = round(float(S1_file.split("_")[7]), 2)
                if lakeModeCheck == S1_lakeModeCheck:
                    print(lakeModeCheck, S1_lakeModeCheck)
                    if selectedYearCheck == S1_selectedYearCheck:
                        if lake_lon == S1_lon:
                            if lake_lat == S1_lat:
                                
                                S1_in = S1_basefolder + S1_file
                                S1_clipped = S1_clippedfolder + lake_file  + '_' + str(lake_lon) + '_' + str(lake_lat) + '.tif'
                                lake_outline = lakes_basefolder + lake_file + '/' + lake_file  + '.shp'
                                # lake_outline = gpd.read_file(lake_outline).set_crs("EPSG:3413").to_crs("EPSG:4326")
                                
                                # gdal.Warp(S1_clipped, S1_in, cutlineDSName=lake_outline, cropToCutline=True)
                                
                                # Read the shapefile as a GeoDataFrame
                                shapefile = gpd.read_file(lake_outline).set_crs("EPSG:3413").to_crs("EPSG:4326")
                                
                                # Reproject the shapefile to match the raster’s CRS, if needed
                                with rasterio.open(S1_in) as src:
                                    if shapefile.crs != src.crs:
                                        shapefile = shapefile.to_crs(src.crs)
                                
                                    # Convert shapefile geometry to GeoJSON format
                                    geoms = [feature["geometry"] for feature in shapefile.__geo_interface__["features"]]
                                
                                    # Crop the raster using the shapefile geometry
                                    out_image, out_transform = mask(src, geoms, crop=True)
                                    out_meta = src.meta.copy()
                                
                                # Update metadata with the new dimensions, transform, and CRS
                                out_meta.update({
                                    "driver": "GTiff",
                                    "height": out_image.shape[1],
                                    "width": out_image.shape[2],
                                    "transform": out_transform
                                })
                                
                                # Write the cropped raster to a new file
                                with rasterio.open(S1_clipped, "w", **out_meta) as dest:
                                    dest.write(out_image)
                                
                                print("Cropping completed successfully.")
                                
                                
            
            # print(file)
            # gdal.Warp(filename_perLake, S1_folder_clipped + filename, cutlineDSName=lake_filename, cropToCutline=True)



#     # Clip Sentinel-1 image to lake shape
#     for file in os.listdir(S1_folder_clipped):
#           filename = os.fsdecode(file)

#           if filename.endswith(".tif"):
             
              # # Make new folder in base-folder
              # new_lake_dir = S1_folder_perLake 
              # os.makedirs(new_lake_dir, exist_ok=True)
             
    #           # Create new filename
    #           filename_perLake = new_lake_dir + '/' + satellite + '_' + selectedYear + '_Lakenumber_' + str(lake_number) + '_' + filename
                                        
    #           # Clip file and save in new folder
    #           gdal.Warp(filename_perLake, S1_folder_clipped + filename, cutlineDSName=lake_filename, cropToCutline=True)
    
    

# for i in range(len(lakes)):
    
#     # Select lake of interest
#     lake_number = str(i)
#     lake_filename = lakes_basefolder + lakeMode + '_' + selectedYear + '_Lakenumber_' + lake_number + '/' + lakeMode + '_' + selectedYear + '_Lakenumber_' + lake_number + '.shp'

#     # Folder with Sentinel-1 data per lake boundary
#     S1_folder_clipped = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1Clips/OttoEtAl/01_RawData/PeriodJuneOct/' + lakeMode + '/'
    
#     # New base-folder with Sentinel-1 data exactly clipped per lake
#     S1_folder_perLake = '/Users/sderodahusman/Documents/PhD/01_Research/06_Data/AI4EO/03_S1Clips/OttoEtAl/02_Clipped/' + lakeMode + '_' + selectedYear + '_Lakenumber_' + lake_number + '/'
  
#     # Clip Sentinel-1 image to lake shape
#     for file in os.listdir(S1_folder_clipped):
#           filename = os.fsdecode(file)

#           if filename.endswith(".tif"):
             
              # # Make new folder in base-folder
              # new_lake_dir = S1_folder_perLake 
              # os.makedirs(new_lake_dir, exist_ok=True)
             
    #           # Create new filename
    #           filename_perLake = new_lake_dir + '/' + satellite + '_' + selectedYear + '_Lakenumber_' + str(lake_number) + '_' + filename
                                        
    #           # Clip file and save in new folder
    #           gdal.Warp(filename_perLake, S1_folder_clipped + filename, cutlineDSName=lake_filename, cropToCutline=True)
    