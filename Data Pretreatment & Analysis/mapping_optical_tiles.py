#################################################################################################################################
# Data Visualization Mapping for Optical tiles
#################################################################################################################################

import os
import numpy as np
from PIL import Image
import cv2
import imgaug.augmenters as iaa
import glob
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from matplotlib.colors import Normalize
import rasterio
import contextily as ctx
from osgeo import gdal
from matplotlib.colors import ListedColormap

# Define the paths to the optical data folder
optical_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet"  # Optical
shapefile_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\global_grid_planet_projlatlon\global_grid_planet_projlatlon.shp"

# List the optical files
optical_files = sorted(glob.glob(os.path.join(optical_folder, "*_merged_modified.tif")))

# Read the projection data
projection_data = gpd.read_file(shapefile_path)

# Define the bounding box for Kalimantan
kalimantan_bbox = box(108.00, -5.00, 120.00, 8.00)

# Filter the projection data to the bounding box
projection_data = projection_data.cx[kalimantan_bbox.bounds[0]:kalimantan_bbox.bounds[2],
                                      kalimantan_bbox.bounds[1]:kalimantan_bbox.bounds[3]]

# Create a base map
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the background map
projection_data.plot(ax=ax, alpha=0.5, color='white', edgecolor='black')

# Get the extent of the Kalimantan bounding box
extent = (kalimantan_bbox.bounds[0], kalimantan_bbox.bounds[2], kalimantan_bbox.bounds[1], kalimantan_bbox.bounds[3])

# Set the extent of the plot to cover the entire Kalimantan area
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])

# Plot optical data on the map as grayscale images
for file_path in optical_files:
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
    ax.imshow(grayscale_image, extent=extent, cmap='terrain', alpha=0.9)

# Customize the map appearance
plt.title("Optical Tiles Mapping")

# Show the map
plt.show()