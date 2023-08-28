# The purpose of this file is to perform visualization of the distribution of validation/training/test samples for NDVI stratified sampling. 

#################################################################################################################################
# Part 1: Importing necessary libraries
#################################################################################################################################

from osgeo import gdal
import os
import glob
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import contextily as ctx
import folium

#################################################################################################################################
# Part 2: Creating Mapping Visualization
#################################################################################################################################

# Define the paths to the input and target data folders
input_folder = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\planet_tiles\Processed Planet" 
target_folder = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\LiDAR\Processed LiDAR"
shapefile_path = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\global_grid_planet_projlatlon\global_grid_planet_projlatlon.shp"

# List the input and target files
input_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
target_files = sorted(glob.glob(os.path.join(target_folder, "*.tif")))

# Calculate NDVI for each input patch and store the mean NDVI values in the ndvi_values array
red_band_indices = [0]  # Assuming the red band is the first band (index 0) in the input patches
nir_band_indices = [3]  # Assuming the near-infrared (NIR) band is the fourth band (index 3) in the input patches
ndvi_values = []
for file_path in input_files:
    dataset = gdal.Open(file_path)
    raster_array = dataset.ReadAsArray()
    red_band = raster_array[red_band_indices]
    nir_band = raster_array[nir_band_indices]
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi_values.append(np.nanmean(ndvi))
    dataset = None  # Close the dataset

# Convert the ndvi_values list to a NumPy array and round to 1 decimal place
ndvi_values = np.array(ndvi_values).round(1)

# Define the threshold for small and large NDVI values
threshold = 0.5

# Create binary labels for small and large NDVI values
labels = (ndvi_values > threshold).astype(int)

# Perform stratified sampling to get one sample for each category in each set
small_indices = np.where(labels == 0)[0]
large_indices = np.where(labels == 1)[0]
train_small_indices, test_small_indices = train_test_split(small_indices, test_size=0.5, random_state=42, stratify=labels[small_indices])
train_large_indices, test_large_indices = train_test_split(large_indices, test_size=0.5, random_state=42, stratify=labels[large_indices])

# Get the corresponding file paths using the indices
train_small_files = [input_files[i] for i in train_small_indices]
test_small_files = [input_files[i] for i in test_small_indices]
train_large_files = [input_files[i] for i in train_large_indices]
test_large_files = [input_files[i] for i in test_large_indices]

# Combine the test sets to form the validation sets
val_small_files, test_small_files = train_test_split(test_small_files, test_size=0.5, random_state=42)
val_large_files, test_large_files = train_test_split(test_large_files, test_size=0.5, random_state=42)

# Combine all sets to get the final training, validation, and test sets
train_files = train_small_files + train_large_files
val_files = val_small_files + val_large_files
test_files = test_small_files + test_large_files

# Print the number of files for each category
print("Number of training files:", len(train_files))
print("Number of validation files:", len(val_files))
print("Number of testing files:", len(test_files))

# Read the projection data
projection_data = gpd.read_file(shapefile_path)

# Define the bounding box for Kalimantan
kalimantan_bbox = box(108.00, -5.00, 120.00, 8.00)

# Filter the projection data to the bounding box
projection_data = projection_data.cx[kalimantan_bbox.bounds[0]:kalimantan_bbox.bounds[2],
                                      kalimantan_bbox.bounds[1]:kalimantan_bbox.bounds[3]]

# Create a custom colormap for each label (training, test, validation)
colors = {'Training Data': 'blue', 'Validation Data': 'green', 'Testing Data': 'red'}
cmap = ListedColormap(list(colors.values()))

# Create a base map
fig, ax = plt.subplots(figsize=(10, 10))

# Initialize an empty list to store scatter points for each label
scatter_points = []

# Plot training, validation, and testing data on the map as raster images
for data, color, label in [(train_files, "blue", "Training Data"), (val_files, "green", "Validation Data"), (test_files, "red", "Testing Data")]:
    for file_path in data:
        # Open the raster image using GDAL
        dataset = gdal.Open(file_path)

        # Read the raster data into a NumPy array
        raster_array = dataset.ReadAsArray()

        # Get the geotransform to determine the extent of the raster image
        geotransform = dataset.GetGeoTransform()

        # Calculate the extent of the raster image
        extent = (geotransform[0],                          # xmin
                  geotransform[0] + geotransform[1] * dataset.RasterXSize,  # xmax
                  geotransform[3] + geotransform[5] * dataset.RasterYSize,  # ymin
                  geotransform[3])                         # ymax

        # Convert the multi-band image to grayscale
        grayscale_image = np.mean(raster_array, axis=0)
        ax.imshow(grayscale_image, extent=extent, cmap=cmap, alpha=0.5, vmin=0, vmax=255)

    # Store scatter points for each label (outside the inner loop)
    scatter_points.append(ax.scatter([], [], c=color, label=label, alpha=0.5))

# Plot the background map
projection_data.plot(ax=ax, alpha=0.2, color='white', edgecolor='black')

# Customize the map appearance
plt.title("NDVI Stratified Sampling Data Mapping")

# Create a single legend for all scatter points
ax.legend(handles=scatter_points)

# Show the map
plt.show()

#################################################################################################################################
# Part 2: Create an interactive map using folium
#################################################################################################################################

# Create the folium map centered on Kalimantan
kalimantan_map = folium.Map(location=[-1.7, 113.4], zoom_start=7)

# Add a basemap layer (OpenStreetMap)
folium.TileLayer(tiles='OpenStreetMap').add_to(kalimantan_map)

# Plot training, validation, and testing data on the map as raster images
for data, color, label in [(train_files, "blue", "Training Data"), (val_files, "green", "Validation Data"), (test_files, "red", "Testing Data")]:
    for file_path in data:
        # Open the raster image using GDAL
        dataset = gdal.Open(file_path)

        # Read the raster data into a NumPy array
        raster_array = dataset.ReadAsArray()

        # Get the geotransform to determine the extent of the raster image
        geotransform = dataset.GetGeoTransform()

        # Calculate the extent of the raster image
        extent = (geotransform[0],                          # xmin
                  geotransform[0] + geotransform[1] * dataset.RasterXSize,  # xmax
                  geotransform[3] + geotransform[5] * dataset.RasterYSize,  # ymin
                  geotransform[3])                         # ymax

        # Convert the multi-band image to grayscale
        grayscale_image = np.mean(raster_array, axis=0)

        # Create a folium image overlay
        image_overlay = folium.raster_layers.ImageOverlay(
            image=grayscale_image,
            bounds=[[extent[1], extent[3]], [extent[0], extent[2]]],
            colormap=cmap,
            opacity=0.5
        )
        image_overlay.add_to(kalimantan_map)

# Add the legend to the map
legend_html = '''
     <div style="position: fixed; bottom: 50px; left: 50px; width: 160px; height: 80px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
     ">&nbsp; <b>Legend</b> <br>
     &nbsp; Training Data &nbsp; <i class="fa fa-circle fa-1x" style="color:blue"></i><br>
     &nbsp; Validation Data &nbsp; <i class="fa fa-circle fa-1x" style="color:green"></i><br>
     &nbsp; Testing Data &nbsp; <i class="fa fa-circle fa-1x" style="color:red"></i>
      </div>
     '''

kalimantan_map.get_root().html.add_child(folium.Element(legend_html))

# Display the interactive map
kalimantan_map
