# The purpose of this file is to visualize the distribution of validation/training/test samples.

#################################################################################################################################
# Part 1: Import necessary libraries
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
# Part 2: Create Mapping Visualization
#################################################################################################################################

# Define the paths to the input and target data folders
input_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet"  
target_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR"  
shapefile_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\global_grid_planet_projlatlon\global_grid_planet_projlatlon.shp"

# List the input and target files
input_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
target_files = sorted(glob.glob(os.path.join(target_folder, "*.tif")))

# Split the files into training, validation, and testing sets (80% for training, 10% for validation, and 10% for testing)
train_files, val_test_files = train_test_split(input_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(val_test_files, test_size=0.5, random_state=42)

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
plt.title("Random Sampling Data Mapping")

# Create a single legend for all scatter points
ax.legend(handles=scatter_points)

# Show the map
plt.show()

#################################################################################################################################
# Part 3: Create an interactive map using folium
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

