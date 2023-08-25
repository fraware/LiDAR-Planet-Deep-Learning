# The objective of this file is to create filters and re-project the Planet files.

#####################################################################################################################
# Creating the cloud masks
#####################################################################################################################

# Importing necessary libraries
import math
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import argparse
from osgeo import gdal
from scipy import ndimage
from PIL import Image, ImageOps
import glob
from skimage.filters import threshold_otsu
from skimage.filters import threshold_triangle
from skimage.filters import threshold_li
import keras
import xarray as xr
import rioxarray as rxr
import rasterio as rio
import random

#####################################################################################################################

# Defining the functions

def get_projection_data(ref_img):
    """
    Get the projection and geotransform data from a reference image.

    Parameters:
        ref_img (str): Path to the reference image.

    Returns:
        str: The projection information.
        tuple: The geotransform information as a tuple (origin_x, pixel_width, 0, origin_y, 0, pixel_height).
    """
    ref_ds = gdal.Open(ref_img)
    projection, geotransform = ref_ds.GetProjection(), ref_ds.GetGeoTransform()
    del ref_ds
    return projection, geotransform

def get_filename(folder, prefix, img_file):
    """
    Generate a new filename based on the folder, prefix, and the original image filename.

    Parameters:
        folder (str): Path to the folder where the new filename will be saved.
        prefix (str): Prefix to be added to the original filename.
        img_file (str): Path to the original image file.

    Returns:
        str: The new filename with the folder and prefix added.
    """
    return os.path.join(folder, prefix.join(os.path.splitext(os.path.split(img_file)[1])))

def find_threshold_value(img_file):
    """
    Find the threshold value for cloud detection based on the pixel value distribution of all bands.

    Parameters:
        img_file (str): Path to the input image file.

    Returns:
        int: The threshold value for cloud detection.
    """
    all_band_values = []

    img_ds = gdal.Open(img_file, gdal.GA_ReadOnly)
    if img_ds is None:
        print(f"Failed to open GDAL dataset for file: {img_file}")
        return None

    # Iterate through all bands and collect pixel values
    for band_index in range(1, img_ds.RasterCount + 1):
        band = img_ds.GetRasterBand(band_index).ReadAsArray()
        all_band_values.extend(band.ravel())

    # # Plot histogram of all bands pixel values
    # plt.figure(figsize=(8, 6))
    # plt.hist(all_band_values, bins=50, range=[0, 255], color='gray', alpha=0.5)
    # plt.title('All Bands Pixel Value Distribution')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.show()

    # Calculate threshold value based on the distribution (e.g., using mean and standard deviation)
    threshold_value = np.mean(all_band_values) + 2 * np.std(all_band_values)

    return int(threshold_value)

def find_threshold_value_otsu(img_file):
    """
    Find the threshold value for cloud detection using Otsu's thresholding method on all bands.

    Parameters:
        img_file (str): Path to the input image file.

    Returns:
        int: The threshold value for cloud detection.
    """
    all_band_values = []

    img_ds = gdal.Open(img_file, gdal.GA_ReadOnly)
    if img_ds is None:
        print(f"Failed to open GDAL dataset for file: {img_file}")
        return None

    # Iterate through all bands and collect pixel values
    for band_index in range(1, img_ds.RasterCount + 1):
        band = img_ds.GetRasterBand(band_index).ReadAsArray()
        all_band_values.extend(band.ravel())

    # Calculate threshold value using Otsu's method on all bands
    threshold_value = threshold_otsu(all_band_values)

    return int(threshold_value)

def find_threshold_value_triangle(img_file):
    """
    Find the threshold value for cloud detection using the Triangle method on all bands.

    Parameters:
        img_file (str): Path to the input image file.

    Returns:
        int: The threshold value for cloud detection.
    """
    all_band_values = []

    img_ds = gdal.Open(img_file, gdal.GA_ReadOnly)
    if img_ds is None:
        print(f"Failed to open GDAL dataset for file: {img_file}")
        return None

    # Iterate through all bands and collect pixel values
    for band_index in range(1, img_ds.RasterCount + 1):
        band = img_ds.GetRasterBand(band_index).ReadAsArray()
        all_band_values.extend(band.ravel())

    # Calculate threshold value using the Triangle method on all bands
    threshold_value = threshold_triangle(all_band_values)

    return int(threshold_value)

def find_threshold_value_li(img_file):
    """
    Find the threshold value for cloud detection using Li's MCE method on all bands.

    Parameters:
        img_file (str): Path to the input image file.

    Returns:
        int: The threshold value for cloud detection.
    """
    all_band_values = []

    img_ds = gdal.Open(img_file, gdal.GA_ReadOnly)
    if img_ds is None:
        print(f"Failed to open GDAL dataset for file: {img_file}")
        return None

    # Iterate through all bands and collect pixel values
    for band_index in range(1, img_ds.RasterCount + 1):
        band = img_ds.GetRasterBand(band_index).ReadAsArray()
        all_band_values.extend(band.ravel())

    # Calculate threshold value using Li's MCE method on all bands
    threshold_value = threshold_li(all_band_values)

    return int(threshold_value)

def save_mask_as_raster_PIL(mask_data, results_folder, mask_name, img_file):
    """
    Save the mask as a 4-band raster dataset.

    Parameters:
        mask_data (numpy.ndarray): The mask data to be saved.
        results_folder (str): Path to the folder where the mask will be saved.
        mask_name (str): Name of the mask (e.g., 'cloud_mask' or 'cloud_shadow_mask').
        img_file (str): Path to the original image file for generating the new filename.
    """
    # Generate the new filename based on the original image filename
    new_filename = get_filename(results_folder, mask_name, img_file)

    # Convert numpy array to PIL Image
    mask_image = Image.fromarray(mask_data)

    # Save the image as a TIFF file
    mask_image.save(new_filename)

def create_cloud_masks(img_file, threshold_value, results_folder):
    """
    Create cloud and cloud shadow masks for a single input image.

    Parameters:
        img_file (str): Path to the input image file.
        threshold_value (int): The threshold value for cloud detection.
        results_folder (str): Path to the folder where the cloud masks will be saved.
    """
    img_ds = gdal.Open(img_file, gdal.GA_ReadOnly)
    if img_ds is None:
        print(f"Failed to open GDAL dataset for file: {img_file}")
        return

    bands, width, height = img_ds.RasterCount, img_ds.RasterXSize, img_ds.RasterYSize
    analytic_stack = np.empty((bands, height, width), dtype=np.uint16)

    for band in range(1, bands + 1):
        banddataraster = img_ds.GetRasterBand(band)
        dataraster = banddataraster.ReadAsArray()
        analytic_stack[band - 1] = dataraster

    img_ds = None  # Close the dataset manually after reading

    for band_index in range(bands):
        _, cloud_mask_band = cv2.threshold(analytic_stack[band_index], threshold_value, 255, cv2.THRESH_BINARY)
        cloud_mask_band = cloud_mask_band.astype(np.uint8) // 255  # Convert thresholded mask to 0 and 1 values
        cloud_shadow_mask_band = np.logical_not(cloud_mask_band)

        # Apply median filter to remove small clumps in the masks
        cloud_mask_band = ndimage.median_filter(cloud_mask_band, size=(3, 3)).astype(np.uint8)
        cloud_shadow_mask_band = ndimage.median_filter(cloud_shadow_mask_band, size=(3, 3)).astype(np.uint8)

        # Apply morphological operations (Closing)
        kernel = np.ones((5, 5), np.uint8)
        cloud_mask_band = cv2.morphologyEx(cloud_mask_band, cv2.MORPH_CLOSE, kernel)
        cloud_shadow_mask_band = cv2.morphologyEx(cloud_shadow_mask_band, cv2.MORPH_CLOSE, kernel)

        # Save the cloud and cloud shadow masks for each band
        band_name = f"_band{band_index + 1}"
        save_mask_as_raster_PIL(cloud_mask_band, results_folder, f"_cloud_mask{band_name}", img_file)
        save_mask_as_raster_PIL(cloud_shadow_mask_band, results_folder, f"_cloud_shadow_mask{band_name}", img_file)

#####################################################################################################################

# Applying the operations to create the masks

# Load the TIFF image
data_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles"
results_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet"

# Process each image in img_files
img_files = [img_file for img_file in glob.glob(os.path.join(data_folder, "*.tif"))]
for img_file in img_files:
    threshold_value = find_threshold_value(img_file)
    #threshold_value = find_threshold_value_otsu(img_file)
    #threshold_value = find_threshold_value_triangle(img_file)
    #threshold_value = find_threshold_value_li(img_file)
    #print("Threshold Value:", threshold_value)
    create_cloud_masks(img_file, threshold_value, results_folder)

#####################################################################################################################
# Applying the masks to the Planet files
#####################################################################################################################

# Importing necessary libraries
import math
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import argparse
from osgeo import gdal
from scipy import ndimage
from PIL import Image, ImageOps
import glob

# # Function to apply mask on the green band of the Planet image and save as 4-channel TIFF (this one was working great!)
# def apply_mask_and_save_as_4_channel_tiff(planet_img_path, cloud_mask_paths, cloud_shadow_mask_paths, masked_value):
#     with rasterio.open(planet_img_path) as planet_ds:
#         # Read all bands from the Planet image
#         num_bands = planet_ds.count
#         band_data = [planet_ds.read(band_index) for band_index in range(1, num_bands + 1)]

#         # Count the number of bands that exceed the threshold
#         num_bands_exceed_threshold = 0
#         for band_index, (cloud_mask_path, cloud_shadow_mask_path) in enumerate(zip(cloud_mask_paths, cloud_shadow_mask_paths)):
#             # Read the cloud mask data
#             with Image.open(cloud_mask_path) as cloud_mask_image:
#                 cloud_mask_data = np.array(cloud_mask_image)  # Convert PIL image to numpy array
#             # Read the cloud shadow mask data
#             with Image.open(cloud_shadow_mask_path) as cloud_shadow_mask_image:
#                 cloud_shadow_mask_data = np.array(cloud_shadow_mask_image)  # Convert PIL image to numpy array

#             # Create a mask where both cloud and cloud shadow areas are masked
#             combined_mask = np.logical_or(cloud_mask_data == 1, cloud_shadow_mask_data == 0)

#             # Check if the threshold is exceeded for the current band
#             if np.all(combined_mask):
#                 num_bands_exceed_threshold += 1

#         # If the threshold is exceeded for at least 2 bands, apply the mask to all bands
#         if num_bands_exceed_threshold >= 2:
#             masked_band_data = []
#             for band_index, (cloud_mask_path, cloud_shadow_mask_path) in enumerate(zip(cloud_mask_paths, cloud_shadow_mask_paths)):
#                 # Read the cloud mask data
#                 with Image.open(cloud_mask_path) as cloud_mask_image:
#                     cloud_mask_data = np.array(cloud_mask_image)  # Convert PIL image to numpy array
#                 # Read the cloud shadow mask data
#                 with Image.open(cloud_shadow_mask_path) as cloud_shadow_mask_image:
#                     cloud_shadow_mask_data = np.array(cloud_shadow_mask_image)  # Convert PIL image to numpy array

#                 # Create a mask where both cloud and cloud shadow areas are masked
#                 combined_mask = np.logical_or(cloud_mask_data == 1, cloud_shadow_mask_data == 0)

#                 # Apply the mask on the current band
#                 masked_band = np.copy(band_data[band_index])
#                 masked_band[combined_mask] = masked_value
#                 masked_band_data.append(masked_band)

#         else:
#             # If the threshold is not exceeded for at least 2 bands, retain the original band data
#             masked_band_data = band_data

#         # Save the modified image as a 4-channel TIFF
#         modified_img_path = os.path.join(processed_masks_folder, f"{os.path.splitext(os.path.basename(planet_img_path))[0]}_modified.tif")
#         with rasterio.open(planet_img_path) as planet_ds:
#             profile = planet_ds.profile  # Get metadata from the original image
#             profile.update(count=num_bands)  # Set the number of bands to the total number of bands
#             with rasterio.open(modified_img_path, 'w', **profile) as modified_ds:
#                 for band_index, band in enumerate(masked_band_data, start=1):
#                     modified_ds.write(band, band_index)


# Function will now check if the threshold is exceeded for both band 1 and band 3 before applying the mask to all bands.
# If the condition is met, the mask will be applied to all bands; otherwise, the original band data will be retained.
def apply_mask_and_save_as_4_channel_tiff(planet_img_path, cloud_mask_path, cloud_shadow_mask_path, masked_value):
    """
    Apply masks from cloud and cloud shadow images to the corresponding bands of the Planet image and save as a 4-channel TIFF.

    Parameters:
        planet_img_path (str): Path to the Planet image file.
        cloud_mask_path (str): Path to the cloud mask image file.
        cloud_shadow_mask_path (str): Path to the cloud shadow mask image file.
        masked_value (int): Value to be used for masked pixels.

    Returns:
        None
    """
    with rasterio.open(planet_img_path) as planet_ds:
        # Read all bands from the Planet image
        num_bands = planet_ds.count
        band_data = [planet_ds.read(band_index) for band_index in range(1, num_bands + 1)]

        # Read the cloud mask data
        with Image.open(cloud_mask_path) as cloud_mask_image:
            cloud_mask_data = np.array(cloud_mask_image)  # Convert PIL image to numpy array

        # Read the cloud shadow mask data
        with Image.open(cloud_shadow_mask_path) as cloud_shadow_mask_image:
            cloud_shadow_mask_data = np.array(cloud_shadow_mask_image)  # Convert PIL image to numpy array

        # Create a mask where both cloud and cloud shadow areas are masked
        combined_mask = np.logical_or(cloud_mask_data == 1, cloud_shadow_mask_data == 0)

        # Apply the mask on all bands at the pixel level
        masked_band_data = []
        for band_index in range(num_bands):
            masked_band = np.copy(band_data[band_index])
            for row in range(masked_band.shape[0]):
                for col in range(masked_band.shape[1]):
                    if combined_mask[row, col]:  # Check if the threshold is exceeded
                        masked_band[row, col] = masked_value
            masked_band_data.append(masked_band)

        # Save the modified image as a 4-channel TIFF
        modified_img_path = os.path.join(processed_masks_folder, f"{os.path.splitext(os.path.basename(planet_img_path))[0]}_modified.tif")
        with rasterio.open(planet_img_path) as planet_ds:
            profile = planet_ds.profile  # Get metadata from the original image
            profile.update(count=num_bands)  # Set the number of bands to the total number of bands
            with rasterio.open(modified_img_path, 'w', **profile) as modified_ds:
                for band_index, band in enumerate(masked_band_data, start=1):
                    modified_ds.write(band, band_index)

#####################################################################################################################

# Applying the function

# Path to the directory containing Planet images and processed masks
planet_data_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles"
processed_masks_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet"

# Get the list of Planet image files
planet_img_files = [img_file for img_file in glob.glob(os.path.join(planet_data_folder, "*.tif"))]

# Value to replace cloud and cloud shadow areas
masked_value = 255

# Iterate through each Planet image and apply the corresponding masks
for planet_img_file in planet_img_files:
    # Create the corresponding mask file names based on the Planet image file
    base_filename = os.path.splitext(os.path.basename(planet_img_file))[0]
    cloud_mask_files = [os.path.join(processed_masks_folder, f"{base_filename}_cloud_mask_band{band_index}.tif") for band_index in range(1, 5)]
    cloud_shadow_mask_files = [os.path.join(processed_masks_folder, f"{base_filename}_cloud_shadow_mask_band{band_index}.tif") for band_index in range(1, 5)]

    # Loop through the mask files and call the function for each pair of masks
    for cloud_mask_file, cloud_shadow_mask_file in zip(cloud_mask_files, cloud_shadow_mask_files):
        # Check if the mask files exist before applying
        if os.path.exists(cloud_mask_file) and os.path.exists(cloud_shadow_mask_file):
            apply_mask_and_save_as_4_channel_tiff(planet_img_file, cloud_mask_file, cloud_shadow_mask_file, masked_value)
        else:
            print(f"Mask files not found for {planet_img_file}. Skipping...")

    # Check if the mask files exist before deleting them
    for mask_file in cloud_mask_files + cloud_shadow_mask_files:
        if os.path.exists(mask_file):
            os.remove(mask_file)
            print(f"Deleted: {mask_file}")


# #####################################################################################################################
# # Checking Aligmnent
# #####################################################################################################################

# import imageio

# def plot_random_input_target(planet_folder, lidar_folder):
#     """
#     Load and plot a random input patch (Optical) and its corresponding target patch (LiDAR).

#     Parameters:
#         planet_folder (str): Path to the folder containing Planet files.
#         lidar_folder (str): Path to the folder containing LiDAR (CHM) files.

#     Returns:
#         None
#     """
#     input_tif_files = os.listdir(planet_folder)
#     target_tif_files = os.listdir(lidar_folder)

#     # Select a random input tif file
#     random_input_file = random.choice(input_tif_files)

#     # Extract common parts from input filename
#     common_parts = "_".join(random_input_file.split("_")[1:4])

#     # Find corresponding target tif file
#     matching_target_files = [
#         file for file in target_tif_files
#         if common_parts in file
#     ]

#     if not matching_target_files:
#         print(f"No matching target file found for input file: {random_input_file}")

#     # Select a random corresponding target tif file
#     random_target_file = random.choice(matching_target_files)

#     # Plot the input and target patches side by side
#     plt.figure(figsize=(10, 5))

#     # Plot the target patch
#     input_path = os.path.join(planet_folder, random_input_file)
#     input_data = imageio.imread(input_path)
#     input_first_band = input_data[:, :, 0]
#     # Plot the input patch (first band)
#     plt.subplot(1, 2, 1)
#     plt.imshow(input_first_band, cmap='gray')  # Use a grayscale colormap
#     plt.title("Input Patch (Optical - First Band)")
#     plt.axis("off")

#     # Plot the target patch
#     plt.subplot(1, 2, 2)
#     target_path = os.path.join(lidar_folder, random_target_file)
#     target_data = plt.imread(target_path)
#     plt.imshow(target_data, cmap="viridis")  # You can choose a different colormap
#     plt.title("Target Patch (LiDAR)")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()

# # Example usage
# planet_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet"
# lidar_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR"
# plot_random_input_target(planet_folder, lidar_folder)
















#----------------------------------------------------------------

# Using Fabien's Pre-trained U-net model

# # Add the model loading and inference code here
# # Use 'with' context managers for both rasterio.open blocks
# with rasterio.open(predicted_path) as src, rasterio.open(reference_path) as reference:
#     # Create the profile for the output raster
#     profile = src.profile
#     profile["BIGTIFF"] = "YES"
#     profile["compress"] = "LZW"
#     profile["count"] = 1
#     profile["tiled"] = True
#     profile["blockysize"] = 128
#     profile["blockxsize"] = 128
#     profile["nodata"] = np.nan

#     # Load your pre-trained model using Keras or PyTorch
#     # Replace 'PATH_TO_MODEL' with the actual path to the model file
#     model = keras.models.load_model(PATH_TO_MODEL, compile=False)

#     # Process each image in img_files
#     img_files = [img_file for img_file in glob.glob(os.path.join(data_folder, "*.tif"))]
#     for img_file in img_files:
#         threshold_value = find_threshold_value(img_file)
#         create_cloud_masks(img_file, threshold_value, results_folder)

#         # Load the subtile data using the 'load_subtile' function
#         granule = "some_granule_name"  # Replace this with the appropriate granule name
#         to_predict, original_size = load_subtile(granule)

#         if to_predict is not None:
#             # Apply the machine learning model to mask clouds
#             prediction = model.predict(to_predict)


# #####################################################################################################################
# # Re-projecting the Planet files at 1m resolution.
# #####################################################################################################################

# # importing the libraries
# import os
# import numpy as np
# import rasterio
# from PIL import Image
# import cv2
# import tifffile as tiff
# import imgaug.augmenters as iaa
# import glob
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, TensorDataset
# import datetime
# import time
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torch.nn.functional as F
# from fastai.vision.all import *
# from torch.utils.data import DataLoader, TensorDataset
# import tifffile
# import dask.array as da
# import imageio
# from scipy.ndimage import zoom
# import tempfile
# import rioxarray
# from rasterio.enums import Resampling
# import tifffile as tiff
# import xarray as xr
# import rioxarray as rxr
# import rasterio as rio
# from skimage.transform import resize
# import os
# import rioxarray as rxr
# import rasterio as rio
# from rasterio.windows import Window

# # Defining the function

# def process_files_in_folder(planet_folder_path, lidar_folder_path, output_folder_path):

#     # Load all chm files from LiDAR folder and sort them
#     chm_data_files = [f for f in os.listdir(lidar_folder_path) if f.endswith(".tif")]
#     chm_data_files.sort()

#     # Sort planet files
#     planet_files = [f for f in os.listdir(planet_folder_path) if f.endswith("_modified.tif")]
#     planet_files.sort()

#     for planet_file, chm_file in zip(planet_files, chm_data_files):
#         planet_file_path = os.path.join(planet_folder_path, planet_file)

#         # Load planet_data using xr.open_dataset
#         planet_data = xr.open_dataset(planet_file_path)

#         # Find the corresponding chm file
#         chm_file_path = os.path.join(lidar_folder_path, chm_file)

#         # Load chm_data using xr.open_dataset
#         chm_data = xr.open_dataset(chm_file_path)

#         # Reproject planet_data to match the projection of chm_data using rasterio
#         reprojected_planet = planet_data.rio.reproject_match(chm_data)

#         # # Print the original and reprojected resolutions
#         # print(f"Original Resolution: {planet_data.rio.resolution()} m")
#         # print(f"Reprojected Resolution: {reprojected_planet.rio.resolution()} m")

#         # # Print the shape of the data array after reprojection
#         # print(f"Reprojected Data Shape: {reprojected_planet.band_data.shape}")

#         # Data Type Conversion
#         try:
#             # Try to save data as int16
#             reprojected_planet.band_data.astype(np.int16).rio.to_raster(f"{output_folder_path}/{os.path.splitext(planet_file)[0]}_reproject.tif")
#             print(f"Data type conversion to np.int16 and saving successful for {planet_file}.")
#         except OverflowError:
#             try:
#                 # If int16 fails, try to save data as float32
#                 reprojected_planet.band_data.astype(np.float32).rio.to_raster(f"{output_folder_path}/{os.path.splitext(planet_file)[0]}_reproject.tif")
#                 print(f"Data type conversion to np.float32 and saving successful for {planet_file}.")
#             except OverflowError:
#                 print(f"Data type conversion to np.int16 and np.float32 failed for {planet_file}.")

#     # Delete files that end with "_modified.tif" in the output folder
#     for file in os.listdir(output_folder_path):
#         if file.endswith("_modified.tif"):
#             file_path = os.path.join(output_folder_path, file)
#             os.remove(file_path)
#             print(f"Deleted {file}.")

# #####################################################################################################################

# # Applying the function

# planet_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet"
# lidar_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR"
# output_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet"

# process_files_in_folder(planet_folder_path, lidar_folder_path, output_folder_path)

#####################################################################################################################

#----- OLD -----


# def process_files_in_folder(planet_folder_path, lidar_folder_path, output_folder_path):

#     # Load all chm files from LiDAR folder and sort them
#     chm_data_files = [f for f in os.listdir(lidar_folder_path) if f.endswith(".tif")]
#     chm_data_files.sort()

#     # Sort planet files
#     planet_files = [f for f in os.listdir(planet_folder_path) if f.endswith("_modified.tif")]
#     planet_files.sort()

#     for planet_file, chm_file in zip(planet_files, chm_data_files):
#         planet_file_path = os.path.join(planet_folder_path, planet_file)

#         # Load planet_data using xr.open_dataset
#         planet_data = xr.open_dataset(planet_file_path)

#         # Find the corresponding chm file
#         chm_file_path = os.path.join(lidar_folder_path, chm_file)

#         # Load chm_data using xr.open_dataset
#         chm_data = xr.open_dataset(chm_file_path)

#         # Reproject planet_data to match the projection of chm_data using rasterio
#         reprojected_planet = planet_data.rio.reproject_match(chm_data)

#         # # Print the original and reprojected resolutions
#         # print(f"Original Resolution: {planet_data.rio.resolution()} m")
#         # print(f"Reprojected Resolution: {reprojected_planet.rio.resolution()} m")

#         # # Print the shape of the data array after reprojection
#         # print(f"Reprojected Data Shape: {reprojected_planet.band_data.shape}")

#         # Data Type Conversion
#         try:
#             # Try to save data as int16
#             reprojected_planet.band_data.astype(np.int16).rio.to_raster(f"{output_folder_path}/{planet_file}_reproject.tif")
#             print(f"Data type conversion to np.int16 and saving successful for {planet_file}.")
#         except OverflowError:
#             try:
#                 # If int16 fails, try to save data as float32
#                 reprojected_planet.band_data.astype(np.float32).rio.to_raster(f"{output_folder_path}/{planet_file}_reproject.tif")
#                 print(f"Data type conversion to np.float32 and saving successful for {planet_file}.")
#             except OverflowError:
#                 print(f"Data type conversion to np.int16 and np.float32 failed for {planet_file}.")

#         # Get the planet_values_array as NumPy array
#         planet_values_array = reprojected_planet.band_data.values

#         # # Reshape the planet_values_array to the desired shape
#         # planet_values_array = planet_values_array.reshape((4, 128, 128, -1))

#         # # Rescale the data down to the desired shape using bilinear interpolation
#         # desired_shape = (4, 128, 128, -1)
#         # planet_values_array = resize(planet_values_array, desired_shape, mode='reflect', anti_aliasing=True)

#         # # Convert NumPy array to PyTorch tensor
#         # planet_values_tensor = torch.from_numpy(planet_values_array)
#         # # Save the planet_values_tensor to a file using torch
#         # output_tensor_file_path = os.path.join(output_folder_path, f"{planet_file}_reprojected.pt")
#         # torch.save(planet_values_tensor, output_tensor_file_path)

#         # Convert and save NumPy array to npy format
#         output_npy_file_path = os.path.join(output_folder_path, f"{os.path.splitext(planet_file)[0]}_reprojected.npy")
#         np.save(output_npy_file_path, planet_values_array)
#         print(f"Data saved as npy format for {planet_file}.")

#     # # Delete files that end with "_reproject.tif" in the output folder
#     # for file in os.listdir(output_folder_path):
#     #     if file.endswith("_reproject.tif"):
#     #         file_path = os.path.join(output_folder_path, file)
#     #         os.remove(file_path)
#     #         print(f"Deleted {file}.")

