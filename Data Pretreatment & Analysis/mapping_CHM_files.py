#################################################################################################################################
# Data Visualization Mapping for LiDAR files
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

# Define the paths to the LiDAR data folder
LiDAR_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\LiDAR"  # Optical
shapefile_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\global_grid_planet_projlatlon\global_grid_planet_projlatlon.shp"

# List the LiDAR files
LiDAR_files = sorted(glob.glob(os.path.join(LiDAR_folder, "*.tif")))

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

# Plot LiDAR data on the map
for file_path in LiDAR_files:
    # Open the raster image using GDAL
    dataset = gdal.Open(file_path)

    # Read the raster data into a NumPy array (elevation data)
    elevation_data = dataset.ReadAsArray()

    # Get the geotransform to determine the extent of the raster data
    geotransform = dataset.GetGeoTransform()

    # Calculate the extent of the raster data
    extent = (geotransform[0],                          # xmin
              geotransform[0] + geotransform[1] * dataset.RasterXSize,  # xmax
              geotransform[3] + geotransform[5] * dataset.RasterYSize,  # ymin
              geotransform[3])                         # ymax

    # Plot the LiDAR data on the map as elevation data (single-band, no grayscale conversion)
    ax.imshow(elevation_data, extent=extent, cmap='terrain', alpha=0.2)

# Customize the map appearance
plt.title("LiDAR Mapping")

# Show the map
plt.show()

#################################################################################################################################
# Height and Gradient Visualization Mapping
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
from osgeo import gdal
import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from matplotlib.colors import ListedColormap
import contextily as ctx
import rasterio
from rasterio.enums import Resampling

# Define the path to the LiDAR CHM data folder
LiDAR_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\LiDAR"  # Optical

# List the CHM files
chm_data_files = sorted(glob.glob(os.path.join(LiDAR_folder, "*.tif")))

# Create a custom colormap for the height (CHM values)
cmap_chm = 'viridis'

# Create a custom colormap for the gradient
cmap_gradient = 'coolwarm'

# Define the bounding box for Kalimantan
kalimantan_bbox = box(108.00, -5.00, 120.00, 8.00)

# Create a base map using contextily (OSM) centered on the bounding box for Kalimantan
fig, (ax_height, ax_gradient) = plt.subplots(1, 2, figsize=(20, 10))
ctx.add_basemap(ax_height, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)
ctx.add_basemap(ax_gradient, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)

# Set the extent for the maps
ax_height.set_xlim(kalimantan_bbox.bounds[0], kalimantan_bbox.bounds[2])
ax_height.set_ylim(kalimantan_bbox.bounds[1], kalimantan_bbox.bounds[3])
ax_gradient.set_xlim(kalimantan_bbox.bounds[0], kalimantan_bbox.bounds[2])
ax_gradient.set_ylim(kalimantan_bbox.bounds[1], kalimantan_bbox.bounds[3])

# Plot the height and gradient on the map
for chm_file in chm_data_files:
    with rasterio.open(chm_file) as dataset:
        # Read the CHM raster data into a NumPy array with downsampling
        chm_data = dataset.read(
            1,
            out_shape=(
                dataset.count,
                int(dataset.height // 4),  # Adjust the downsample factor as needed
                int(dataset.width // 4)   # Adjust the downsample factor as needed
            ),
            resampling=Resampling.average
        )

        # Get the geotransform to determine the extent of the raster image
        geotransform = dataset.transform

        # Calculate the extent of the raster image
        extent = (geotransform[0],                          # xmin
                  geotransform[0] + geotransform[1] * dataset.width,  # xmax
                  geotransform[3] + geotransform[5] * dataset.height,  # ymin
                  geotransform[3])                         # ymax

        # Plot height (CHM values)
        ax_height.imshow(chm_data, extent=extent, cmap=cmap_chm, alpha=0.5)

        # Calculate the gradient
        gradient_x, gradient_y = np.gradient(chm_data)
        gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # Plot gradient
        ax_gradient.imshow(gradient, extent=extent, cmap=cmap_gradient, alpha=0.5)

# Customize the map appearance for height (CHM values)
ax_height.set_title("LiDAR CHM Height Map")
ax_height.set_xlabel("Longitude")
ax_height.set_ylabel("Latitude")
ax_height.set_aspect('equal')

# Customize the map appearance for gradient
ax_gradient.set_title("LiDAR CHM Gradient Map")
ax_gradient.set_xlabel("Longitude")
ax_gradient.set_ylabel("Latitude")
ax_gradient.set_aspect('equal')

# Show the maps
plt.show()

#################################################################################################################################
