# The purpose of this file is to perform a Biomass Analysis based on our predicted wall-to-wall map

#################################################################################################################################
# Part 1: Importing necessary libraries
#################################################################################################################################

import os
import numpy as np
import rasterio
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import datashader as ds
import datashader.transfer_functions as tf
import skimage.util as sk_util
import plotly.express as px
from skimage.transform import resize
import tifffile as tiff
import imgaug.augmenters as iaa
import glob
import torch
import pyproj
from rasterio.warp import calculate_default_transform, reproject, Resampling
import itertools
from rasterio import windows
import pandas as pd
import holoviews as hv
import hvplot.pandas
torch.cuda.empty_cache()
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_tensor
import datetime
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from imagecodecs import imwrite, imread
import torchvision.models as models
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

print("PyTorch version:", torch.__version__)
from fastai.vision.all import *
from torchsummary import summary
import tifffile
import dask.array as da
import imageio
from skimage import exposure, transform
from scipy.ndimage import zoom
from scipy import ndimage
import tempfile
import rioxarray
from rasterio.enums import Resampling
import tifffile as tiff
import xarray as xr
import rioxarray as rxr
import rasterio as rio
import torch.optim as optim
from affine import Affine
from tqdm import tqdm
from pyproj import Transformer
from osgeo import gdal


#################################################################################################################################
# Part 2: Compute and save Kalimantan_percent_cover.tif for each 1-hectare (1-ha) grid cell
#################################################################################################################################

# Define the paths to the data folders
wall_to_wall_map_file = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Output\wall_to_wall_map_kalimantan.tif"

# Open the wall-to-wall map file with rasterio
with rasterio.open(wall_to_wall_map_file) as src:
    # Read the wall-to-wall map data as a numpy array
    wall_to_wall_map = src.read(1)  # Assuming it's a single-band image

    # Define the threshold height in meters
    threshold = 5  # Threshold height in meters

    # Create a mask to exclude NaN values from the computation
    mask = ~np.isnan(wall_to_wall_map)

    # Calculate percent canopy cover for each 1-ha grid cell
    cell_size = 100  # 1 ha = 100 m x 100 m
    percent_cover = np.zeros(
        (
            wall_to_wall_map.shape[0] // cell_size,
            wall_to_wall_map.shape[1] // cell_size,
        ),
        dtype=np.float32,
    )

    for row in range(0, wall_to_wall_map.shape[0], cell_size):
        for col in range(0, wall_to_wall_map.shape[1], cell_size):
            cell = wall_to_wall_map[row : row + cell_size, col : col + cell_size]
            valid_pixels = cell[cell > threshold]
            if valid_pixels.size > 0:
                percent_cover[row // cell_size, col // cell_size] = (
                    np.count_nonzero(valid_pixels) / valid_pixels.size
                ) * 100

    # Define the output file path for the percent canopy cover map
    output_file = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\Kalimantan_percent_cover_1ha.tif"

    # Get the metadata from the source file
    meta = src.meta.copy()

    # Update the data type to float32 for the percent cover
    meta["dtype"] = "float32"
    x_origin, y_origin = (
        meta["transform"][2],
        meta["transform"][5],
    )  # Origin coordinates

    # Update the transform with the correct 1-ha grid cell resolution
    meta["transform"] = rasterio.transform.from_origin(
        x_origin, y_origin, cell_size, cell_size
    )  # Assuming 1-ha grid with cell_size m resolution

    # Create a new GeoTIFF file to save the percent canopy cover
    with rasterio.open(output_file, "w", **meta) as dst:
        # Write the percent canopy cover data to the new file
        dst.write(percent_cover, 1)

    # Visualize the generated percent canopy cover map using a suitable visualization library
    plt.imshow(wall_to_wall_map, cmap="viridis")
    plt.colorbar(label="Percent Canopy Cover (%)")
    plt.title("Percent Canopy Cover Map")
    plt.show()

# Print the calculated percent canopy cover
print(f"Percent Canopy Cover: {percent_cover:.2f}%")

#################################################################################################################################
# Part 3: Compute and save Kalimantan_LCA.tif for each 1-hectare (1-ha) grid cell
#################################################################################################################################

# To compute the Large Crown Area (LCA) within each 1-ha grid cell using the LCA algorithm, we'll follow the steps described:
# Apply a threshold of 27 meters on the 1-m CHM to derive a binary image representing the areas covered by high vegetation (large trees).
# Perform connected component segmentation with 8-neighborhood pixel connectivity to compute clusters of pixels representing large tree areas.
# Remove clusters composed of less than 100 pixels (i.e., 100 m²) to get the LCA binary image at 1-m resolution.
# Resample the LCA binary image to a 1-ha grid to represent the percentage of area covered by large trees within each 1-ha cell.

# Open the wall-to-wall map file with rasterio
with rasterio.open(wall_to_wall_map_file) as src:
    # Read the wall-to-wall map data as a numpy array
    wall_to_wall_map = src.read(1)

    # Define the threshold height in meters
    lca_threshold = 27  # Threshold height in meters

    # Create a binary image representing high vegetation areas (large trees)
    binary_lca = wall_to_wall_map > lca_threshold

    # Perform connected component segmentation to compute clusters of large tree areas
    lca_clusters, num_clusters = ndimage.label(binary_lca, structure=np.ones((3, 3)))

    # Remove clusters smaller than 100 pixels (100 m²)
    min_cluster_size = 100
    for cluster_label in range(1, num_clusters + 1):
        cluster_size = np.sum(lca_clusters == cluster_label)
        if cluster_size < min_cluster_size:
            lca_clusters[lca_clusters == cluster_label] = 0

    # Convert the large tree clusters to 1-ha grid by summing up the pixels within each 1-ha cell
    cell_size = 100  # Assuming 1-ha cell size in meters
    lca_1ha_grid = np.zeros(
        (wall_to_wall_map.shape[0] // cell_size, wall_to_wall_map.shape[1] // cell_size)
    )
    for row in range(0, wall_to_wall_map.shape[0], cell_size):
        for col in range(0, wall_to_wall_map.shape[1], cell_size):
            cell = lca_clusters[row : row + cell_size, col : col + cell_size]
            lca_1ha_grid[row // cell_size, col // cell_size] = np.sum(cell > 0)

    # Compute the percentage of area covered by large trees for each 1-ha cell
    lca_percent_cover = (lca_1ha_grid / (cell_size**2)) * 100

    # Define the output file path for the Large Crown Area (LCA) percent cover map
    output_file_lca = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\Kalimantan_LCA.tif"

    # Get the metadata from the source file
    meta = src.meta.copy()

    # Update the data type to float32 for the LCA percent cover
    meta["dtype"] = "float32"

    # Create a new GeoTIFF file to save the LCA percent cover
    with rasterio.open(output_file_lca, "w", **meta) as dst:
        # Write the LCA percent cover data to the new file
        dst.write(lca_percent_cover, 1)

    # Visualize the generated Large Crown Area (LCA) percent cover map using a suitable visualization library
    plt.imshow(lca_percent_cover, cmap="viridis")
    plt.colorbar(label="LCA Percent Cover (%)")
    plt.title("Large Crown Area (LCA) Percent Cover Map")
    plt.show()

# Print the calculated Large Crown Area (LCA) percent cover
print(f"Large Crown Area (LCA) Percent Cover: {lca_percent_cover:.2f}%")

#################################################################################################################################
# Part 4: Compute and save Kalimantan_MCH.tif for each 1-hectare (1-ha) grid cell
#################################################################################################################################

# Load the wall-to-wall tree height map
with rasterio.open(wall_to_wall_map_file) as src:
    tree_height_map = src.read(1)
    transform = src.transform  # Get the georeferencing transform

# Define the cell size (1 ha) in meters
cell_size = 100  # 1 ha = 100 m x 100 m

# Initialize an empty array to store MCH values
mch_values = np.zeros(
    (tree_height_map.shape[0] // cell_size, tree_height_map.shape[1] // cell_size)
)

# Iterate through the grid cells and calculate MCH for each
for row in range(0, tree_height_map.shape[0], cell_size):
    for col in range(0, tree_height_map.shape[1], cell_size):
        cell = tree_height_map[row : row + cell_size, col : col + cell_size]
        valid_pixels = cell[
            cell > 0
        ]  # Exclude zero values (assuming no negative heights)
        if valid_pixels.size > 0:
            mch = np.mean(valid_pixels)
            mch_values[row // cell_size, col // cell_size] = mch

# Define the output file path for the MCH GeoTIFF
output_file_mch = (
    r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\Kalimantan_MCH.tif"
)

# Save the MCH values to a new GeoTIFF file
height, width = mch_values.shape
with rasterio.open(
    output_file_mch,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,  # MCH is a single-band raster
    dtype=mch_values.dtype,
    crs=src.crs,  # Use the same CRS as the input tree height map
    transform=transform,  # Use the same georeferencing transform
) as dst:
    dst.write(mch_values, 1)

# Visualize the MCH values as a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(mch_values, cmap="viridis", extent=src.bounds, origin="upper")
plt.colorbar(label="Mean Canopy Height (MCH) in meters")
plt.title("Mean Canopy Height (MCH) Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

#################################################################################################################################
# Part 5: Compute and save Kalimantan_degradation_index.tif for each 1-hectare (1-ha) grid cell
#################################################################################################################################

# Calculate the Forest Degradation Index (FDI) as FDI = MCH + LCA + PC
fdi = mch_values + lca_percent_cover + percent_cover

# Save the Forest Degradation Index (FDI) map to a new GeoTIFF file (Kalimantan_degradation_index.tif)
output_file_fdi = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 80%\Kalimantan_degradation_index.tif"

# Classify FDI values into user-defined degradation severity classes
# You can define the class thresholds as per your requirement
intact_threshold = 255
light_degraded_threshold = 235
moderate_degraded_threshold = 210
high_degraded_threshold = 150

fdi_classes = np.zeros_like(fdi, dtype=np.uint8)
fdi_classes[fdi > intact_threshold] = 1
fdi_classes[(fdi > light_degraded_threshold) & (fdi <= intact_threshold)] = 2
fdi_classes[(fdi > moderate_degraded_threshold) & (fdi <= light_degraded_threshold)] = 3
fdi_classes[(fdi > high_degraded_threshold) & (fdi <= moderate_degraded_threshold)] = 4
fdi_classes[fdi <= high_degraded_threshold] = 5

# Get the metadata from the wall-to-wall map file
with rasterio.open(wall_to_wall_map_file) as src:
    meta = src.meta.copy()

# Update the data type to uint8 for the FDI classes
meta["dtype"] = "uint8"

# Create a new GeoTIFF file to save the FDI map
with rasterio.open(output_file_fdi, "w", **meta) as dst:
    # Write the FDI classes data to the new file
    dst.write(fdi_classes, 1)

# Visualize the generated Forest Degradation Index (FDI) map using a suitable visualization library
plt.imshow(fdi_classes, cmap="viridis")
plt.colorbar(label="Forest Degradation Index (FDI)")
plt.title("Forest Degradation Index (FDI) Map")
plt.show()
