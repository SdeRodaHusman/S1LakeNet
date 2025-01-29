import ee

# Authenticate to the Earth Engine API
ee.Initialize()


# Load your Feature Collection with Bounding Boxes
filepath = "projects/ee-sophiederoda/assets/AI4EO/HydrofracturingBBox/"
# year = 2021
# filename = "HydrofracturingEvents_" + str(year) + "_Shackleton_Sommer"

# year = 2020
# filename = "HydrofracturingEvents_" + str(year) + "_Amery_Trusel"
# ########################################
year = 2022
filename = "HydrofracturingEvents_All_Amery_Trusel"
# Define the date range
start_date = str(year-1) + '-12-01'
end_date = str(year) + '-03-01'
# ########################################
# year = 2016
# filename = "HydrofracturingEvents_" + str(year) + "_Greenland_BenedekWillis"
# start_date = str(year-1) + '-11-01'
# end_date = str(year) + '-03-01'
# ########################################
# year = 2019
# filename = "HydrofracturingEvents_" + str(year) + "_Greenland_Poinar"
# start_date = str(year) + '-06-01'
# end_date = str(year) + '-09-01'
# ########################################
bbox = ee.FeatureCollection(filepath + filename)
bbox_col = bbox.toList(bbox.size());
# ########################################



bbox_n = bbox.size().getInfo()

for i in range(0,bbox_n,1):
    bbox_i = bbox_col.get(i)
    bbox_geometry = ee.Feature(bbox_i).geometry()
    
    S2_col = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(bbox_geometry) \
                .filterDate(ee.Date(start_date), ee.Date(end_date)) \
                .select(['B4','B3','B2','QA60']) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))

    S2_n = S2_col.size().getInfo()
    S2_list = S2_col.toList(S2_col.size());

    for j in range(0,S2_n,1):
        S2_img = ee.Image(S2_list.get(j))
        S2_clip = S2_img.clip(bbox_geometry)

        # Remove cloudy images    
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        
        qa60 = S2_img.select(['QA60'])
        
        cloudMask = qa60.bitwiseAnd(cloudBitMask).eq(0).And(qa60.bitwiseAnd(cirrusBitMask).eq(0))
        
        # Reduce the region. The region parameter is the Feature geometry.
        cloudPercentageRegion = cloudMask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=bbox_geometry,
            scale=30,
            maxPixels=1e9)
        
        if cloudPercentageRegion.get('QA60').getInfo() > 0.9:
        
            # Get time of image to use as image name
            overpassTime = S2_img.get('system:time_start').getInfo()
            centroid = bbox_geometry.centroid()
            coordinates = centroid.coordinates().getInfo()
            latitude = coordinates[1]
            longitude = coordinates[0]
    
            export_filename = 'Optical_clipped_' + filename + '_' + str(latitude) + '_' + str(longitude) + '_' + str(overpassTime)
            print('Export image: ', export_filename)
            
        
              # Export the image to an Earth Engine asset.
            task = ee.batch.Export.image.toCloudStorage(**{
              'image': S2_img.select(['B4','B3','B2']).divide(1e4).clip(bbox_geometry),
              'bucket': 'ee-surfacemelt',
              'description': filename,
              'fileNamePrefix': 'S1_Hydrofracturing/GEE_exports/Optical/TruselAllYears/' + export_filename + '_S2',
              # 'scale': 10,
              'dimensions': '201x201',
              'region': bbox_geometry.getInfo()['coordinates']})
            task.start()

for i in range(0,bbox_n,1):
    bbox_i = bbox_col.get(i)
    bbox_geometry = ee.Feature(bbox_i).geometry()
    
    L8_col = ee.ImageCollection('LANDSAT/LC08/C02/T2_TOA') \
                .filterBounds(bbox_geometry) \
                .filterDate(ee.Date(start_date), ee.Date(end_date)) \
                .select(['B4','B3','B2','QA_PIXEL']) \
                .filter(ee.Filter.lt('CLOUD_COVER', 50))

    L8_n = L8_col.size().getInfo()
    L8_list = L8_col.toList(L8_col.size());
    
    for j in range(0,L8_n,1):
        L8_img = ee.Image(L8_list.get(j))
        L8_clip = L8_img.clip(bbox_geometry)

        # Remove cloudy pixels
        cloudShadowBitMask = 1 << 3;
        cloudsBitMask = 1 << 5;
        
        cloudMask = L8_img.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
        #.bitwiseAnd(cloudShadowBitMask).eq(0)
        #cloudMask = mask.bitwiseAnd(cloudsBitMask).eq(0)
        
        
        # cloudBitMask = 1 << 10  # Landsat 8 uses the QA band bit 5 for cloud masking
        # cloudMask = L8_img.select(['QA_PIXEL']).bitwiseAnd(cloudBitMask).eq(0)

        # Reduce the region. The region parameter is the Feature geometry.
        cloudPercentageRegion = cloudMask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=bbox_geometry,
            scale=30,
            maxPixels=1e9)
        print(cloudPercentageRegion.getInfo())
        if cloudPercentageRegion.get('QA_PIXEL').getInfo() > 0.9:
            # Get time of image to use as image name
            overpassTime = L8_img.get('system:time_start').getInfo()
            centroid = bbox_geometry.centroid()
            coordinates = centroid.coordinates().getInfo()
            latitude = coordinates[1]
            longitude = coordinates[0]
    
            export_filename = 'Optical_clipped_' + filename + '_' + str(latitude) + '_' + str(longitude) + '_' + str(overpassTime)
            print('Export image: ', export_filename)
            
            
              # Export the image to an Earth Engine asset.
            task = ee.batch.Export.image.toCloudStorage(**{
              'image': L8_img.select(['B4','B3','B2']).clip(bbox_geometry),
              'bucket': 'ee-surfacemelt',
              'description': filename,
              'fileNamePrefix': 'S1_Hydrofracturing/GEE_exports/Optical/TruselAllYears/' + export_filename + '_L8',
              # 'scale': 10,
              'dimensions': '201x201',
              'region': bbox_geometry.getInfo()['coordinates']})
            task.start()
        else:
            print('I am not going to export this image - there are too many clouds!')
        
