#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:52:11 2024

@author: sderodahusman
"""


import ee
ee.Initialize()

# # ---------------------------------------------------------------------
# # Settings:
# # ---------------------------------------------------------------------
nan_tiles = [2, 7, 8, 9, 16, 17, 20, 21, 26, 33, 41, 42, 75, 88, 97, 104, 108, 126, 129, 130, 133, 134, 135, 138, 155, 160, 161, 162, 167, 176]

# to do: 57,58,59,60
for tile_number in range(57,61):
# for tile_number in range(3,4):

    if tile_number in nan_tiles:# or tile_number == 7:
        continue  
    
    year = 2021
    tile = tile_number # 149
    satellite = 'S2'    
    
    # Define the date range
    start_date = str(year-1) + '-12-01'
    end_date = str(year) + '-03-01'
     
    folder = 'projects/ee-sophiederoda/assets/AI4EO/LakeMasks/' + str(year) + '/'
    
    # Load the gridTiles_iceShelves FeatureCollection
    gridTiles_iceShelves = ee.FeatureCollection('projects/ee-earthmapps/assets/gridTiles_iceShelves')
    
    # Select the tile of interest
    selectedTile = gridTiles_iceShelves.toList(gridTiles_iceShelves.size()).get(ee.Number.parse(str(tile)))
    selectedTile = ee.Feature(selectedTile).geometry()
    
    # Define the filename
    filename = folder + 'tile-' + str(tile) + '_' + satellite + '_' + str(year) + '-01-01_' + str(year) + '-01-31_' + 'maxlake_maxvolume'
    
    # Load the image
    img = ee.Image(filename)
    
    # Vectorize lakes
    lakes = img.select('LakeMask_max').reduceToVectors(**{
        'reducer': ee.Reducer.countEvery(),
        'geometry': selectedTile,
        'scale': 10,
        'crs': 'EPSG:3031',
        'maxPixels': 1e13})
    
    # Select lakes larger than 500,000 m2
    largeLakes = lakes.filter(ee.Filter.gte('count', 500))
    
    if largeLakes.size().getInfo() == 0:
        continue
    
    # Create list of collection
    collectionList = largeLakes.toList(largeLakes.size());
    n = collectionList.size().getInfo();
    print('Tile of interest: ', tile, '|||| Number of images to upload to GEE:', n)
    
      
    # ---------------------------------------------------------------------
    #  Import Sentinel-1:
    # ---------------------------------------------------------------------
    
    # Buffer lakes
    boundingBoxSize = 1000
    
    # Function to compute the number of unmasked pixels in an image
    def compute_unmasked_pixels(image):
        # Count the number of unmasked pixels in the image
        pixel_count = image.reduceRegion(
            reducer=ee.Reducer.count(), 
            geometry=bbox_geometry, 
            scale=10,  # Assuming the pixel resolution is 10 meters
            maxPixels=1e9  # Maximum number of pixels to reduce
        ).get('HH')
        
        # Add the pixel count as a property to the image
        return image.set('unmasked_pixel_count', pixel_count)
    
    # Export images to Asset
    # for i in range(73, n, 1):
    for i in range(0, n, 1):
        geometry = ee.Feature(collectionList.get(i)).geometry()    
        buffered_geometry = geometry.buffer(boundingBoxSize)
        centroid = buffered_geometry.centroid()
        coordinates = centroid.coordinates().getInfo()
        latitude = coordinates[1]
        longitude = coordinates[0]
        bounding_box = centroid.buffer(boundingBoxSize / 2).bounds()
        bbox_geometry = ee.Feature(bounding_box).geometry()
    
        # Compute volume and area
        area = ee.Feature(collectionList.get(i)).get('count').getInfo() * 100
        volume = img.select('VOLUME_max').toInt().reduceRegion(**{
            'reducer':ee.Reducer.sum().setOutputs(["volume_m3"]),
            'geometry':geometry,
            'scale':10}).get('VOLUME_max').getInfo()
        
        # Select lakes with volumes larger than 100,000 m3
        if volume < 100000:
            continue
  
        # Import Sentinel-1
        S1_col = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'HH')) \
            .select('HH') \
            .filterDate(start_date, end_date) \
            .filterBounds(bbox_geometry) \
            .map(compute_unmasked_pixels) \
            .filter(ee.Filter.eq('resolution_meters', 10)) \
            .limit(75) 
            
        # Only select images that are not party masked
        S1_col = S1_col.filter(ee.Filter.gt('unmasked_pixel_count', 9000)); # At least 90% is unmasked
    
        # Sort the image collection by acquisition time
        S1_col= S1_col.sort('system:time_start');
    
        # Get unique orbit numbers
        unique_orbits = S1_col.aggregate_array('relativeOrbitNumber_start').distinct()
        print(unique_orbits.getInfo())
    
        # Iterate over each unique orbit number
        for orbit in unique_orbits.getInfo():
            
            S1_per_orbit = S1_col.filterMetadata('relativeOrbitNumber_start', 'equals', orbit)
            
            # Check size 
            S1_size = S1_per_orbit.size().getInfo() 
            print('Length of timeseries: ', S1_size)
            
            # Convert the sorted image collection to bands
            S1_img = S1_per_orbit.toBands()  
               
            export_filename = 'S1_clipped_' + str(year) + '_' + str(tile)+ '_' + str(latitude) + '_' + str(longitude) + '_' + str(orbit) + '_Area_' + str(area) + '_Volume_' + str(volume)
            print('Lake number: ', i, '|| Exported image: ', export_filename)
                               
            # Export the image to an Earth Engine asset.
            task = ee.batch.Export.image.toCloudStorage(**{
              'image': S1_img.multiply(1e8).toInt32().clip(bbox_geometry),
              'bucket': 'ee-surfacemelt',
              'description': export_filename,
              'fileNamePrefix': 'S1_Hydrofracturing/GEE_exports/S1_AISwide/' + str(year) + '/' + export_filename,
              'crs': 'epsg:4326',
              'dimensions': '100x100',
              'region': bbox_geometry.getInfo()['coordinates']})
            task.start()



