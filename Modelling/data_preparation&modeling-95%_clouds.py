#################################################################################################################################
# Data Preparation and Modelling for Random Sampling
#################################################################################################################################

#################################################################################################################################
# Data preparation
#################################################################################################################################

import os
import numpy as np
import rasterio
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import skimage.util as sk_util
import plotly.express as px
from skimage.transform import resize
import tifffile as tiff
import imgaug.augmenters as iaa
import glob
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_tensor
import datetime
import time
import shutil
from imagecodecs import imwrite, imread
import torch.nn as nn
import torchvision.models as models
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
)
print("PyTorch version:", torch.__version__) # PyTorch version: 2.0.1+cu118
import torch.nn.functional as F
from fastai.vision.all import *
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import tifffile
from skimage.transform import resize
import dask.array as da
import imageio
from skimage import exposure, transform
from scipy.ndimage import zoom
import tempfile
import rioxarray
import datashader as ds
import datashader.transfer_functions as tf
import seaborn as sns
from rasterio.enums import Resampling
import tifffile as tiff
import xarray as xr
import rioxarray as rxr
import rasterio as rio
import torch.optim as optim
import os
import rioxarray as rxr
import rasterio as rio
from rasterio.windows import Window
from torch.utils.tensorboard import SummaryWriter
from skimage.util import view_as_windows
import os
import numpy as np
import rasterio
from PIL import Image
import cv2
import tifffile as tiff
import imgaug.augmenters as iaa
import glob
from rasterio import windows
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import datetime
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models as models
import torch.nn.functional as F
from fastai.vision.all import *
from torch.utils.data import DataLoader, TensorDataset
import tifffile
import dask.array as da
import imageio
from scipy.ndimage import zoom
import tempfile
import rioxarray
from rasterio.enums import Resampling
import tifffile as tiff
import xarray as xr
import rioxarray as rxr
import rasterio as rio
from skimage.transform import resize
import os
import rioxarray as rxr
import rasterio as rio
from rasterio.windows import Window
from tqdm import tqdm
import rasterio.crs as rcrs
from rasterio.crs import CRS
import xarray as xr
import concurrent.futures

# Define the paths to the input and target data folders
input_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet"  # Optical
target_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR"  # LiDAR

# Load Data and extract Patches

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    # Set the device to CUDA (GPU)
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(device))
else:
    print("CUDA is not available.")
    device = torch.device("cpu")

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    print("Number of available GPUs:", num_gpus)
else:
    print("No GPUs available.")

# Compute the percentage of valid (non-NaN) pixels in the data
def calculate_valid_pixel_percentage_NaN(data):
    """
    Calculate the percentage of valid (non-NaN) pixels in the data.

    Parameters:
        data (numpy.ndarray): Input data as a NumPy array.

    Returns:
        float: Percentage of valid (non-NaN) pixels in the data.
    """
    total_pixels = data.size
    valid_pixels = np.count_nonzero(~np.isnan(data))
    valid_pixel_percentage = (valid_pixels / total_pixels) * 100

    return valid_pixel_percentage

# Compute the percentage of valid pixels (pixels not equal to 255) in the data
def calculate_valid_pixel_percentage_255(data):
    """
    Calculate the percentage of valid pixels (pixels not less than 254) in the data.

    Parameters:
        data (numpy.ndarray): Input data as a NumPy array.

    Returns:
        float: Percentage of valid pixels (pixels not less than 254) in the data.
    """
    total_pixels = data.size
    valid_pixels = np.count_nonzero(data <= 253)
    valid_pixel_percentage = (valid_pixels / total_pixels) * 100

    return valid_pixel_percentage

# def determine_patch_size(folder_path, min_patch_size, max_patch_size, step=1):
#     best_patch_size = None
#     best_valid_pixel_percentage = 0
#     best_num_patches = 0

#     for patch_size in range(min_patch_size, max_patch_size *2, step):
#         total_valid_pixel_percentage = 0
#         total_num_patches = 0
#         total_files = 0

#         for filename in os.listdir(folder_path):
#             if filename.endswith(".tif"):
#                 filepath = os.path.join(folder_path, filename)
#                 with rasterio.open(filepath) as src:
#                     # Read the CHM data
#                     data = src.read(1, masked=True)

#                     # Calculate the percentage of valid pixels in the patch
#                     valid_pixel_percentage = calculate_valid_pixel_percentage(data[:patch_size, :patch_size])

#                     # Calculate the number of patches that can be extracted from the file
#                     num_patches = (data.shape[0] // patch_size) * (data.shape[1] // patch_size)

#                     total_valid_pixel_percentage += valid_pixel_percentage
#                     total_num_patches += num_patches
#                     total_files += 1

#         # Calculate the average percentage of valid pixels for the current patch size
#         average_valid_pixel_percentage = total_valid_pixel_percentage / total_files

#         # Check if the current patch size meets the 80% criterion and maximizes the number of patches
#         if average_valid_pixel_percentage >= 80 and total_num_patches > best_num_patches:
#             best_patch_size = patch_size
#             best_valid_pixel_percentage = average_valid_pixel_percentage
#             best_num_patches = total_num_patches

#     return best_patch_size, best_valid_pixel_percentage, best_num_patches


# # Computing the optimal patch size # takes more than 662 min to run
# min_patch_size = 256
# max_patch_size = 1024
# step = 10
# best_patch_size, valid_pixel_percentage, best_num_patches = determine_patch_size(target_folder, min_patch_size, max_patch_size, step)
# print("Best patch size:", best_patch_size) #64
# print("Valid pixel percentage in the patch:", valid_pixel_percentage) #99.98739053914835
# print("Best number of patches:", best_num_patches) #3948912

# # Choose the patch size for the optical data
# patch_size = best_patch_size
patch_size = 256

def normalize_target_chm(target_images):
    """
    Normalize the target images by calculating the vegetation height and scaling it to [0, 1].

    Parameters:
        target_images (list of numpy.ndarray): List of target images as NumPy arrays.

    Returns:
        list of numpy.ndarray: List of normalized target images.
    """
    normalized_target_images = np.empty_like(target_images, dtype=np.float32)

    for idx, target_image in enumerate(target_images):
        # Extract non-NaN data from the target image
        valid_indices = ~np.isnan(target_image)
        valid_data = target_image[valid_indices]

        if len(valid_data) == 0:
            # If there is no valid data, use the original image as is
            normalized_target_image = target_image
        else:
            # Calculate the vegetation height above zero (ground elevation assumed to be zero) for valid data
            vegetation_height = np.maximum(valid_data, 0)

            # Normalize the vegetation height to [0, 1]
            normalized_vegetation_height = vegetation_height / vegetation_height.max()

            # Fill the valid data with normalized vegetation height, and keep NaNs as is
            normalized_target_image = np.empty_like(target_image, dtype=np.float32)
            normalized_target_image[valid_indices] = normalized_vegetation_height
            normalized_target_image[np.isnan(target_image)] = np.nan

        normalized_target_images[idx] = normalized_target_image

    return normalized_target_images

def normalize_band_old(band):
    """
    Normalize a single band of image data to the range [0, 1].

    Parameters:
        band (numpy.ndarray): Input 2D array representing a single band of image data.

    Returns:
        numpy.ndarray: Normalized band with pixel values in the range [0, 1].
    """
    band_min = np.min(band)
    band_max = np.max(band)

    # Check for division by zero or when band_min and band_max are equal
    if band_max - band_min == 0:
        print("Warning: Division by zero or equal band_min and band_max")
        # Handle the case when all pixel values in the band are the same
        normalized_band = np.ones_like(band, dtype=np.float32) * 1e-6
    else:
        normalized_band = (band - band_min) / (band_max - band_min)

    return normalized_band

def normalize_band(band):
    """
    Normalize a single band of image data to the range [0, 1] by dividing by 255.

    Parameters:
        band (numpy.ndarray): Input 2D array representing a single band of image data.

    Returns:
        numpy.ndarray: Normalized band with pixel values in the range [0, 1].
    """
    normalized_band = band / 255.0
    return normalized_band

def normalize_input(optical_images):
    """
    Normalize pixel values to [0, 1] for each band in the optical images.

    Parameters:
        optical_images (numpy.ndarray): Input array of optical image data to be normalized.

    Returns:
        numpy.ndarray: Array of normalized optical images with pixel values between 0 and 1.
    """
    try:
        num_bands, height, width = optical_images.shape
        normalized_optical_images = np.empty_like(optical_images, dtype=np.float32)

        for j in range(num_bands):
            normalized_band = normalize_band(optical_images[j])  # Access each band directly
            normalized_optical_images[j] = normalized_band

    except ValueError as e:
        print(f"Error normalizing images: {str(e)}")
        return None

    return normalized_optical_images

def generate_data_window(src, patch_size, overlap):
    """
    Generate data windows from the source raster with overlapping tiles.

    Parameters:
        src (rasterio.io.DatasetReader): The source rasterio DatasetReader object.
        patch_size (int): Size of the patches to extract from the source raster.
        overlap (int): The size of overlap between tiles.

    Yields:
        rasterio.windows.Window: A data window for each tile.
    """
    raster_window = windows.Window(
        col_off=0, row_off=0, width=src.meta["width"], height=src.meta["height"]
    )
    offsets = itertools.product(
        range(0, src.meta["width"], patch_size - overlap),
        range(0, src.meta["height"], patch_size - overlap),
    )
    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=patch_size,
            height=patch_size,
        )
        yield window.intersection(raster_window)

def process_files_in_folder(planet_file_paths, chm_file_paths):
    """
    Reproject and save the Planet files to match the projection of LiDAR files.

    Parameters:
        planet_file_paths (list): List of file paths to the Planet files.
        chm_file_paths (list): List of file paths to the LiDAR (CHM) files.

    Returns:
        None
    """
    reprojected_files = []  # Store paths of reprojected files

    for planet_file_path, chm_file_path in zip(planet_file_paths, chm_file_paths):
        try:
            with xr.open_dataset(planet_file_path) as planet_data, xr.open_dataset(chm_file_path) as chm_data:

                # Check if the datasets have valid data
                if planet_data.band_data.size == 0 or chm_data.band_data.size == 0:
                    raise ValueError("Empty data in one of the datasets.")

                # Reproject planet_data to match the projection of chm_data using rasterio
                reprojected_planet = planet_data.rio.reproject_match(chm_data)

                # Data Type Conversion
                try:
                    # Try to save data as int16
                    new_filename = os.path.splitext(planet_file_path)[0] + "_reprojected.tif"
                    reprojected_planet.band_data.astype(np.uint8).rio.to_raster(new_filename)
                    print(f"Data type conversion to np.uint8 and saving successful for {new_filename}.")
                    reprojected_files.append(new_filename)  # Add the reprojected file path to the list
                except OverflowError:
                    try:
                        # If int16 fails, try to save data as float32
                        new_filename = os.path.splitext(planet_file_path)[0] + "_reprojected.tif"
                        reprojected_planet.band_data.astype(np.float32).rio.to_raster(new_filename)
                        print(f"Data type conversion to np.float32 and saving successful for {new_filename}.")
                        reprojected_files.append(new_filename)  # Add the reprojected file path to the list
                    except OverflowError:
                        print(f"Data type conversion to np.uint8 and np.float32 failed for {planet_file_path}.")

                # Close the reprojected file
                reprojected_planet.close()

        except Exception as e:
            print(f"Error processing files {planet_file_path} and {chm_file_path}: {str(e)}")

    # Add a delay to ensure that any file handles are released
    time.sleep(2)

    # Delete the original files after all files have been reprojected, closed, and renamed
    for original_file in planet_file_paths:
        if os.path.exists(original_file):
            os.remove(original_file)

    # Rename the newly created files to original names without the "_reprojected" suffix
    for reprojected_file in reprojected_files:
        original_file = reprojected_file.replace("_reprojected", "")
        os.rename(reprojected_file, original_file)

def process_and_save_input_patches(input_folder, patch_size, valid_pixel_threshold=95.0, overlap=0):
    """
    Process and save input patches from the input images in the specified folder.

    Parameters:
        input_folder (str): Path to the input folder containing 'input_patches' subfolder.
        patch_size (int): Size of the patches to extract from the input images.
        valid_pixel_threshold (float): Minimum percentage of valid pixels required in a patch.
        overlap (int, optional): The size of overlap between patches. Default is 0.

    Returns:
        None
    """
    # Get the list of input file paths
    input_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))

    # Get the list of chm file paths
    chm_files = sorted(glob.glob(os.path.join(target_folder, "*.tif")))

    # Create a folder to save the patches if it doesn't exist
    patches_folder = os.path.join(input_folder, "input_patches")
    os.makedirs(patches_folder, exist_ok=True)

    # Re-project and save the Planet files to match the projection of LiDAR files
    planet_file_paths = [file_path for file_path in input_files]
    chm_file_paths = [file_path for file_path in chm_files]
    process_files_in_folder(planet_file_paths, chm_file_paths)

    for input_file in input_files:
        input_path = os.path.join(input_folder, input_file)
        # Open the input image using rasterio
        with rasterio.open(input_path) as input_ds:
            input_image = input_ds.read()

            # Get the data windows for input image
            windows_list = [w for w in generate_data_window(input_ds, patch_size, overlap)]

            # Process each data window and save patches
            def process(w):
                patch = input_image[:, w.row_off : w.row_off + patch_size, w.col_off : w.col_off + patch_size]
                valid_pixel_percentage = calculate_valid_pixel_percentage_255(patch)

                if valid_pixel_percentage >= valid_pixel_threshold:
                    # Convert the patch data type to uint8
                    patch = (patch).astype(np.uint8)

                    # Save the patch as an npy file
                    patch_filename = os.path.join(
                        patches_folder,
                        f"{os.path.splitext(os.path.basename(input_file))[0]}_patch_{w.row_off}_{w.col_off}.npy",
                    )
                    np.save(patch_filename, patch)

            print(f"Processing {os.path.basename(input_file)}")
            # Use ThreadPoolExecutor for concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                list(tqdm(executor.map(process, windows_list), total=len(windows_list)))

def process_and_save_target_patches(target_folder, patch_size, overlap=0):
    """
    Process and save target patches from the target images in the specified folder.

    Parameters:
        target_folder (str): Path to the target folder containing target images.
        patch_size (int): Size of the patches to extract from the target images.
        overlap (int, optional): The size of overlap between patches. Default is 0.

    Returns:
        None
    """
    # Get the list of target file paths
    target_files = sorted(glob.glob(os.path.join(target_folder, "*.tif")))

    # Create a folder to save the target patches if it doesn't exist
    patches_folder = os.path.join(target_folder, "target_patches")
    os.makedirs(patches_folder, exist_ok=True)

    for target_file in target_files:
        target_path = os.path.join(target_folder, target_file)
        # Open the target image using rasterio
        with rasterio.open(target_path) as target_ds:
            target_image = target_ds.read(1)
            # Mask values less than 0 to np.nan
            target_image[target_image < 0] = np.nan

            # Get the data windows for target image
            windows_list = [w for w in generate_data_window(target_ds, patch_size, overlap)]

            # Process each data window and save patches
            def process(w):
                patch = target_image[w.row_off : w.row_off + patch_size, w.col_off : w.col_off + patch_size]
                valid_pixel_percentage = calculate_valid_pixel_percentage_NaN(patch)

                # Save the patch as a separate file if it has enough valid pixels
                if valid_pixel_percentage >= 80.0:
                    # Convert the patch data type to float32
                    patch = patch.astype(np.float32)

                    # Save the patch as an npy file
                    patch_filename = os.path.join(
                        patches_folder,
                        f"{os.path.splitext(os.path.basename(target_file))[0]}_patch_{w.row_off}_{w.col_off}.npy",
                    )
                    np.save(patch_filename, patch)

            print(f"Processing {os.path.basename(target_file)}")
            # Use ThreadPoolExecutor for concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                list(tqdm(executor.map(process, windows_list), total=len(windows_list)))

def find_matching_pairs(input_folder, target_folder):
    """
    Find matching pairs of input and target files in the specified folders.

    Parameters:
        input_folder (str): Path to the input folder containing 'input_patches' subfolder.
        target_folder (str): Path to the target folder containing 'target_patches' subfolder.

    Returns:
        list: A list of matching pairs, where each pair is a tuple containing the paths of input and target files.
    """
    input_files = sorted(glob.glob(os.path.join(input_folder, "input_patches_within_95", "*.npy")))
    target_files = sorted(glob.glob(os.path.join(target_folder, "target_patches_within_95", "*.npy")))

    matching_pairs = []

    for target_file in target_files:
        # Extract the common pattern and numbers from the target file name
        target_pattern = re.findall(r'Polygon_(\d+_utm_\d+\w+)_patch_(\d+_\d+)', target_file)
        if not target_pattern:
            print("no match found for target_pattern")
            continue

        target_num_1, target_num_2 = target_pattern[0]

        for input_file in input_files:
            # Extract the common pattern and numbers from the input file name
            input_pattern = re.findall(r'Polygon_(\d+_utm_\d+\w+)_merged_modified_patch_(\d+_\d+)', input_file)
            if not input_pattern:
                continue

            input_num_1, input_num_2 = input_pattern[0]

            # Check if the extracted pattern and numbers match
            if input_num_1 == target_num_1 and input_num_2 == target_num_2:
                matching_pairs.append((input_file, target_file))
                break

    # # Remove unmatched input files
    # unmatched_input_files = set(input_files) - set(pair[0] for pair in matching_pairs)
    # for unmatched_input_file in unmatched_input_files:
    #     try:
    #         os.remove(unmatched_input_file)
    #     except Exception as e:
    #         print(f"Error removing unmatched input file {os.path.basename(unmatched_input_file)}: {str(e)}")

    # # Remove unmatched target files
    # unmatched_target_files = set(target_files) - set(pair[1] for pair in matching_pairs)
    # for unmatched_target_file in unmatched_target_files:
    #     try:
    #         os.remove(unmatched_target_file)
    #     except Exception as e:
    #         print(f"Error removing unmatched target file {os.path.basename(unmatched_target_file)}: {str(e)}")

    return matching_pairs

def calculate_ndvi(red_band, nir_band):
    """
    Calculate the Normalized Difference Vegetation Index (NDVI).

    Parameters:
        red_band (np.array): Array representing the red band.
        nir_band (np.array): Array representing the near-infrared (NIR) band.

    Returns:
        np.array: The calculated NDVI array.
    """
    # Perform NDVI calculation
    ndvi = (nir_band - red_band) / (nir_band + red_band)

    return ndvi

def load_and_preprocess_data(planet_folder_path, lidar_folder_path, output_folder):
    """
    Load and preprocess the data from the specified folders and split it into training, validation, and test sets.

    Parameters:
        output_folder (str): Path to the output folder to save the preprocessed data.

    Returns:
        None
    """

    # Keeping only the matching between input and target data
    matching_pairs = find_matching_pairs(input_folder, target_folder)

    # Extract the corresponding file paths from the matching_pairs
    planet_file_paths = [os.path.join(planet_folder_path, planet_file) for planet_file, _ in matching_pairs]
    chm_file_paths = [os.path.join(lidar_folder_path, chm_file) for _, chm_file in matching_pairs]

    # Load, normalize, and store target patches
    normalized_target_patches = []
    for target_file_path in chm_file_paths:
        target_patches = np.load(target_file_path, mmap_mode='r')
        if target_patches.shape != (256, 256):
            continue  # Skip patches with incorrect target shape
        normalized_target_patches.append(normalize_target_chm(target_patches))

    # Load, normalize, and store input patches
    normalized_input_patches = []
    for input_file_path in planet_file_paths:
        input_patches = np.load(input_file_path, mmap_mode='r')
        if input_patches.shape != (4, 256, 256):
            continue  # Skip patches with incorrect input shape
        normalized_input_patches.append(normalize_input(input_patches))

    # Convert the lists to NumPy arrays
    normalized_target_patches_array = np.array(normalized_target_patches)
    normalized_input_patches_array = np.array(normalized_input_patches)

    # Rounding the data
    rounded_input_patches_array = np.round(normalized_input_patches_array, decimals=3)
    rounded_target_patches_array = np.round(normalized_target_patches_array, decimals=3)

    # Option 1: Random Sampling Method

    # Split the data into training, validation, and test sets (80% for training, 10% for validation, and 10% for testing)
    train_input, test_input, train_target, test_target = train_test_split(
        rounded_input_patches_array, rounded_target_patches_array, test_size=0.2
    ) #, random_state=42, shuffle=True

    train_input, val_input, train_target, val_target = train_test_split(
        train_input, train_target, test_size=0.5
    ) #, random_state=42, shuffle=True

    # # Option 2: NDVI Stratified Sampling Method

    # # Calculate NDVI for each input patch
    # red_band_indices = [0]  # Assuming the red band is the first band (index 0) in the input patches
    # nir_band_indices = [3]  # Assuming the near-infrared (NIR) band is the fourth band (index 3) in the input patches
    # ndvi_values = []
    # for patch in input_patches:
    #     red_band = patch[..., red_band_indices]
    #     nir_band = patch[..., nir_band_indices]
    #     ndvi = calculate_ndvi(red_band, nir_band)
    #     ndvi_values.append(np.mean(ndvi))  # Store the mean NDVI value for each patch

    # # Convert the lists to NumPy arrays and round to 1 decimal place
    # input_patches = np.array(input_patches)
    # target_patches = np.array(target_patches)
    # ndvi_values = np.array(ndvi_values)

    # Define the threshold for small and large NDVI values
    # threshold = 0.5

    # # Create binary labels for small and large NDVI values
    # labels = (ndvi_values > threshold).astype(int)

    # # Perform stratified sampling to get one sample for each category in each set
    # small_indices = np.where(labels == 0)[0]
    # large_indices = np.where(labels == 1)[0]
    # train_small_indices, test_small_indices = train_test_split(small_indices, test_size=0.5, random_state=42, stratify=labels[small_indices])
    # train_large_indices, test_large_indices = train_test_split(large_indices, test_size=0.5, random_state=42, stratify=labels[large_indices])

    # # Get the corresponding file paths using the indices
    # train_small_files = [input_files[i] for i in train_small_indices]
    # test_small_files = [input_files[i] for i in test_small_indices]
    # train_large_files = [input_files[i] for i in train_large_indices]
    # test_large_files = [input_files[i] for i in test_large_indices]

    # # Combine the test sets to form the validation sets
    # val_small_files, test_small_files = train_test_split(test_small_files, test_size=0.5, random_state=42)
    # val_large_files, test_large_files = train_test_split(test_large_files, test_size=0.5, random_state=42)

    # # Combine all sets to get the final training, validation, and test sets
    # train_files = train_small_files + train_large_files
    # val_files = val_small_files + val_large_files
    # test_files = test_small_files + test_large_files

    # # Saving the output
    # train_input = input_patches[train_input_indices]
    # val_input = input_patches[val_input_indices]
    # test_input = input_patches[test_input_indices]
    # train_target = target_patches[train_target_indices]
    # val_target = target_patches[val_target_indices]
    # test_target = target_patches[test_target_indices]

    # Save the arrays to .npy files
    os.makedirs(output_folder, exist_ok=True)
    train_input_path = os.path.join(output_folder, "train_input.npy")
    val_input_path = os.path.join(output_folder, "val_input.npy")
    test_input_path = os.path.join(output_folder, "test_input.npy")
    train_target_path = os.path.join(output_folder, "train_target.npy")
    val_target_path = os.path.join(output_folder, "val_target.npy")
    test_target_path = os.path.join(output_folder, "test_target.npy")

    np.save(train_input_path, train_input)
    np.save(val_input_path, val_input)
    np.save(test_input_path, test_input)
    np.save(train_target_path, train_target)
    np.save(val_target_path, val_target)
    np.save(test_target_path, test_target)

    # Print the number of samples in each set
    print(f"Shape of training input samples: {train_input.shape}")
    print(f"Shape of validation input samples: {val_input.shape}")
    print(f"Shape of test input samples: {test_input.shape}")
    print(f"Shape of training target samples: {train_target.shape}")
    print(f"Shape of validation target samples: {val_target.shape}")
    print(f"Shape of test target samples: {test_target.shape}")
    print("#########################################################")
    print(f"Number of training input samples: {train_input.shape[0]}")
    print(f"Number of validation input samples: {val_input.shape[0]}")
    print(f"Number of test input samples: {test_input.shape[0]}")
    print(f"Number of training target samples: {train_target.shape[0]}")
    print(f"Number of validation target samples: {val_target.shape[0]}")
    print(f"Number of test target samples: {test_target.shape[0]}")

def convert_to_tensors(train_input_path, val_input_path, test_input_path, train_target_path, val_target_path, test_target_path):
    """
    Load the data from .npy files and convert it to PyTorch tensors.

    Parameters:
        train_input_path (str): Path to the .npy file containing training input patches.
        val_input_path (str): Path to the .npy file containing validation input patches.
        test_input_path (str): Path to the .npy file containing test input patches.
        train_target_path (str): Path to the .npy file containing training target patches.
        val_target_path (str): Path to the .npy file containing validation target patches.
        test_target_path (str): Path to the .npy file containing test target patches.

    Returns:
        tuple: A tuple containing PyTorch tensors for train input, validation input, test input,
               train target, validation target, and test target patches.
    """

    # Load the data from .npy files
    train_input_patches = np.load(train_input_path)
    val_input_patches = np.load(val_input_path)
    test_input_patches = np.load(test_input_path)
    train_target_patches = np.load(train_target_path)
    val_target_patches = np.load(val_target_path)
    test_target_patches = np.load(test_target_path)

    # Convert the numpy data to PyTorch tensors
    train_input_patches = torch.from_numpy(train_input_patches)
    val_input_patches = torch.from_numpy(val_input_patches)
    test_input_patches = torch.from_numpy(test_input_patches)
    train_target_patches = torch.from_numpy(train_target_patches)
    val_target_patches = torch.from_numpy(val_target_patches)
    test_target_patches = torch.from_numpy(test_target_patches)

    # Move data to the appropriate device
    train_input_patches = train_input_patches.to(device)
    val_input_patches = val_input_patches.to(device)
    test_input_patches = test_input_patches.to(device)
    train_target_patches = train_target_patches.to(device)
    val_target_patches = val_target_patches.to(device)
    test_target_patches = test_target_patches.to(device)

    return train_input_patches, val_input_patches, test_input_patches, train_target_patches, val_target_patches, test_target_patches

# Treating the input data
process_and_save_input_patches(input_folder, patch_size)

# Treating the target data
process_and_save_target_patches(target_folder, patch_size)

# Splitting and saving the input data the target data
planet_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet\input_patches_within_95"
lidar_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\target_patches_within_95"
output_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%"
load_and_preprocess_data(planet_folder_path, lidar_folder_path, output_folder)

# Converting the data to tensors
output_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%"
train_input_path = os.path.join(output_folder, "train_input.npy")
val_input_path = os.path.join(output_folder, "val_input.npy")
test_input_path = os.path.join(output_folder, "test_input.npy")
train_target_path = os.path.join(output_folder, "train_target.npy")
val_target_path = os.path.join(output_folder, "val_target.npy")
test_target_path = os.path.join(output_folder, "test_target.npy")

train_input_patches, val_input_patches, test_input_patches, train_target_patches, val_target_patches, test_target_patches = \
    convert_to_tensors(train_input_path, val_input_path, test_input_path, train_target_path, val_target_path, test_target_path)

train_target_patches_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\train_target_patches.pth"
torch.save(train_target_patches, train_target_patches_path)
print("Smoothed train input data saved successfully.")

#################################################################################################################################
# Data Validation Tests
#################################################################################################################################

# # -------------------------------------------------------------------------------------------------------------
# # Checking for NaNs in tensors
# # -------------------------------------------------------------------------------------------------------------

# def has_nan(tensor):
#     return torch.isnan(tensor).any()

# if has_nan(test_target_patches):
#     print("Tensor contains NaN values")
# else:
#     print("Tensor does not contain NaN values")

# # My chm data: train_target_patches, val_target_patches, and test_target_patches contains NaN values.

# def nan_percentage(tensor):
#     nan_count = torch.isnan(tensor).sum().item()
#     total_elements = tensor.numel()
#     return (nan_count / total_elements) * 100.0

# # Example usage
# percentage = nan_percentage(train_target_patches)
# print(f"Percentage of NaN values: {percentage:.2f}%")

# # Percentage of NaN values for train_target_patches: 2.16%
# # Percentage of NaN values for val_target_patches: 2.44%
# # Percentage of NaN values for train_target_patches: 2.16%

# # -------------------------------------------------------------------------------------------------------------
# # Checking for NaNs in target patches
# # -------------------------------------------------------------------------------------------------------------

# folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\target_patches_within_95"
# total_npy_files = 0
# npy_files_with_nan = 0

# for filename in os.listdir(folder_path):
#     if filename.endswith(".npy"):
#         total_npy_files += 1
#         file_path = os.path.join(folder_path, filename)
#         data = np.load(file_path)
#         if np.isnan(data).any():
#             npy_files_with_nan += 1

# print(f"Total number of .npy files in the folder: {total_npy_files}")
# print(f"Number of .npy files with at least one NaN value: {npy_files_with_nan}")

# # Total number of .npy files in the folder: 14791
# # Number of .npy files with at least one NaN value: 11470

# # -------------------------------------------------------------------------------------------------------------
# # Number of bands Tests
# # -------------------------------------------------------------------------------------------------------------

# def get_number_of_bands(tiff_file_path):
#     with rasterio.open(tiff_file_path) as dataset:
#         num_bands = dataset.count  # Get the number of bands
#     return num_bands

# # Path to TIFF file
# tiff_file_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet\Polygon_002_utm_50N_merged_modified.tif"
# num_bands = get_number_of_bands(tiff_file_path)
# print(f"Number of bands in the TIFF file: {num_bands}")


# def check_number_of_bands(npy_file_path):
#     try:
#         npy_data = np.load(npy_file_path)
#         num_bands = npy_data.shape[0]
#         print(f"Number of bands in {npy_file_path}: {num_bands}")
#     except Exception as e:
#         print(f"Error loading or checking bands: {str(e)}")

# # Path to .npy file
# npy_file_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet\input_patches_within_95\Polygon_001_utm_50S_merged_modified_patch_768_4096.npy"
# check_number_of_bands(npy_file_path)


# def check_npy_shapes(folder_path):
#     npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

#     for npy_file in npy_files:
#         npy_path = os.path.join(folder_path, npy_file)
#         npy_array = np.load(npy_path)
#         print(f"Shape of {npy_file}: {npy_array.shape}")

# # Path to .npy files
# #folder_path = r'C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\target_patches_within_95'
# folder_path = r'C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet\input_patches_within_95'
# check_npy_shapes(folder_path)

# def check_npy_shapes(folder_path):
#     npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
#     all_shapes_match = True

#     for npy_file in npy_files:
#         npy_path = os.path.join(folder_path, npy_file)
#         npy_array = np.load(npy_path)
#         print(f"Shape of {npy_file}: {npy_array.shape}")
        
#         if npy_array.shape != (256, 256):
#             all_shapes_match = False

#     if all_shapes_match:
#         print("All files have the correct shape.")
#     else:
#         print("Not all files have the correct shape.")

# # Path to .npy files
# #folder_path = r'C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\target_patches_within_95'
# folder_path = r'C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet\input_patches_within_95'
# check_npy_shapes(folder_path)


# def normalize_single_input(optical_image):
#     """
#     Normalize pixel values to [0, 255] for each band in a single optical image.

#     Parameters:
#         optical_image (numpy.ndarray): The optical image data to be normalized.

#     Returns:
#         numpy.ndarray: Normalized optical image with pixel values between 0 and 255.
#     """
#     npy_data = np.load(optical_image)
#     normalized_optical_image = np.empty_like(npy_data, dtype=np.uint8)
#     try:
#         # Normalize pixel values to [0, 255] for each band in the optical image
#         num_bands = npy_data.shape[0]
#         for i in range(num_bands):
#             normalized_band = normalize_band(npy_data[i])
#             normalized_optical_image[i] = normalized_band

#     except IndexError as e:
#         print(f"Error normalizing image: {str(e)}")

#     return normalized_optical_image

# # Provide the path to your .npy file
# npy_file_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet\input_patches_within_95\Polygon_001_utm_50S_merged_modified_patch_768_4096.npy"
# normalize_single_input(npy_file_path)

# # -------------------------------------------------------------------------------------------------------------
# # Image Comparison Tests
# # -------------------------------------------------------------------------------------------------------------

# # Paths to npy files
# planet_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet\input_patches_within_95"
# lidar_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\target_patches_within_95"
# input_path = os.path.join(planet_folder_path, "Polygon_001_utm_50S_merged_modified_patch_768_4096.npy")
# target_path = os.path.join(lidar_folder_path, "Polygon_001_utm_50S_patch_768_4096.npy")

# # Get a list of all npy files in the folder
# input_npy_files = [f for f in os.listdir(planet_folder_path) if f.endswith(".npy")]
# target_npy_files = [f for f in os.listdir(lidar_folder_path) if f.endswith(".npy")]

# # Select a random input npy file
# random_input_file = random.choice(input_npy_files)

# # Extract common parts from input filename
# input_parts = random_input_file.split("_")
# common_parts = "_".join(input_parts[1:4])

# # Find corresponding target npy file
# matching_target_files = [
#     file for file in target_npy_files
#     if common_parts in file and input_parts[-2:] == file.split("_")[-2:]
# ]

# if not matching_target_files:
#     print(f"No matching target file found for input file: {random_input_file}")

# # Select a random corresponding target npy file
# random_target_file = random.choice(matching_target_files)

# # Load the random npy files
# input_path = os.path.join(planet_folder_path, random_input_file)
# target_path = os.path.join(lidar_folder_path, random_target_file)

# input_data = np.load(input_path)
# target_data = np.load(target_path)

# # Plot the input and target patches side by side
# plt.figure(figsize=(10, 5))

# # Plot the input patch
# plt.subplot(1, 2, 1)
# plt.imshow(np.moveaxis(input_data[:-1],0,2))
# plt.title("Input Patch (Optical)")
# plt.axis("off")

# # Plot the target patch
# plt.subplot(1, 2, 2)
# plt.imshow(target_data, cmap="viridis")  # You can choose a different colormap
# plt.title("Target Patch (LiDAR)")
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# #--------------------------------------------------------------------------------------------------------------
# # Study of the relationship between the height and NDVI
# # -------------------------------------------------------------------------------------------------------------

# def calculate_ndvi_and_heights(input_file, target_file):
#     """
#     Calculate NDVI values and heights for each pixel in the input patch.

#     Parameters:
#         input_file (str): Path to the input patch.
#         target_file (str): Path to the CHM LiDAR data.

#     Returns:
#         ndvi_values (list): List of NDVI values for each pixel.
#         height_values (list): List of height values for each pixel.
#     """
#     ndvi_values = []
#     height_values = []

#     # Load the input patch data and CHM LiDAR data
#     input_patch = np.load(input_file)
#     chm_data = np.load(target_file)

#     # Assuming bands 2 and 3 are Red and NIR respectively (0-based indexing)
#     red_band = input_patch[2, :, :]
#     nir_band = input_patch[3, :, :]

#     # Calculate NDVI and compute height for each pixel
#     for row in range(red_band.shape[0]):
#         for col in range(red_band.shape[1]):
#             red_value = red_band[row, col]
#             nir_value = nir_band[row, col]
#             chm_height = chm_data[row, col]

#             if not np.isnan(chm_height) and red_value < 253 and nir_value < 253: # Check for valid CHM height
#                 ndvi = (nir_value - red_value) / (nir_value + red_value + 1e-9)
#                 ndvi_values.append(ndvi)
#                 height_values.append(chm_height)

#     return ndvi_values, height_values

# # Paths to the input patch and CHM LiDAR data
# input_path = os.path.join(planet_folder_path, "Polygon_001_utm_50S_merged_modified_patch_1280_1536.npy")
# target_path = os.path.join(lidar_folder_path, "Polygon_001_utm_50S_patch_1280_1536.npy")

# # Calculate NDVI values and heights for each pixel
# ndvi_values, height_values = calculate_ndvi_and_heights(input_path, target_path)

# # Calculate the correlation coefficient between NDVI values and heights
# correlation_coefficient = np.corrcoef(ndvi_values, height_values)[0, 1]
# print("Correlation Coefficient:", correlation_coefficient)

# # Plot the relationship between NDVI and heights
# plt.scatter(ndvi_values, height_values, alpha=0.2)
# plt.xlabel("NDVI")
# plt.ylabel("Height")
# plt.title("Relationship between NDVI and Height (Pixel Level)")
# plt.grid()
# plt.show()

# # Correlation Heatmap
# correlation_matrix = np.corrcoef(np.array([ndvi_values, height_values]))
# sns.set(font_scale=1)
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=["NDVI", "Mean Height"], yticklabels=["NDVI", "Mean Height"])
# plt.title("Correlation Heatmap")
# plt.show()

# # Regression Line Plot
# plt.figure(figsize=(10, 6))
# sns.regplot(x=ndvi_values, y=height_values, scatter_kws={'s': 20})
# plt.xlabel("NDVI")
# plt.ylabel("Mean Height")
# plt.title("Regression Line Plot: NDVI vs Mean Height")
# plt.grid()
# plt.show()

# # Joint Distribution Plot
# sns.set(font_scale=1.2)
# sns.jointplot(x=ndvi_values, y=height_values, kind="scatter", height=7)
# plt.show()

# # Violin Plot by NDVI Bin
# df = pd.DataFrame({"NDVI": ndvi_values, "Mean Height": height_values})
# sns.violinplot(x="NDVI", y="Mean Height", data=df)
# plt.xlabel("NDVI")
# plt.ylabel("Mean Height")
# plt.title("Violin Plot by NDVI Bin")
# plt.show()

# # Box Plot by NDVI Bin
# sns.boxplot(x=ndvi_values, y=height_values)
# plt.xlabel("NDVI")
# plt.ylabel("Mean Height")
# plt.title("Box Plot by NDVI Bin")
# plt.show()

# # Hexbin Plot
# plt.figure(figsize=(10, 6))
# plt.hexbin(ndvi_values, height_values, gridsize=20, cmap="YlOrRd")
# plt.xlabel("NDVI")
# plt.ylabel("Mean Height")
# plt.title("Hexbin Plot: NDVI vs Mean Height")
# plt.colorbar(label="Density")
# plt.show()

# # 3D Scatter Plot
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ndvi_values, height_values, range(len(height_values)), c='b', marker='o')
# ax.set_xlabel('NDVI')
# ax.set_ylabel('Mean Height')
# ax.set_zlabel('Data Point')
# ax.set_title("3D Scatter Plot: NDVI vs Mean Height vs Data Point")
# plt.show()

# #----------------------------------------------------------------------------------------------------------------------------
# # At this point, we check the number of sample that do not have clouds in entry for Planet data compared to with clouds
# # ---------------------------------------------------------------------------------------------------------------------------

# # Code to count the number of files within the threshold range
# planet_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet\input_patches_within_95"

# files_with_valid_pixels_99 = 0
# files_with_valid_pixels_95 = 0
# files_with_valid_pixels_80 = 0

# for filename in os.listdir(planet_folder_path):
#     if filename.endswith(".npy"):
#         file_path = os.path.join(planet_folder_path, filename)
#         data = np.load(file_path)

#         valid_pixel_mask = data <= 253
#         valid_pixel_ratio = np.count_nonzero(valid_pixel_mask) / np.prod(data.shape)

#         if valid_pixel_ratio >= 0.99:
#             files_with_valid_pixels_99 += 1

#         if valid_pixel_ratio >= 0.95:
#             files_with_valid_pixels_95 += 1

#         if valid_pixel_ratio >= 0.80:
#             files_with_valid_pixels_80 += 1

# print(f"Number of files with at least 99% valid pixels: {files_with_valid_pixels_99}")
# print(f"Number of files with at least 95% valid pixels: {files_with_valid_pixels_95}")
# print(f"Number of files with at least 80% valid pixels: {files_with_valid_pixels_80}")

# # Number of files with at least 99% valid pixels: 6807
# # Number of files with at least 95% valid pixels: 9979
# # Number of files with at least 80% valid pixels: 14791

# # Code to display a random file within the threshold range
# valid_files = []

# for filename in os.listdir(planet_folder_path):
#     if filename.endswith(".npy"):
#         file_path = os.path.join(planet_folder_path, filename)
#         data = np.load(file_path)

#         valid_pixel_mask = data <= 253
#         valid_pixel_ratio = np.count_nonzero(valid_pixel_mask) / np.prod(data.shape)

#         if valid_pixel_ratio >= 0.95:
#             valid_files.append(file_path)

# if valid_files:
#     selected_file = random.choice(valid_files)
#     selected_data = np.load(selected_file)
#     plt.imshow(selected_data[1])  # Assuming you want to plot the first band
#     plt.title(f"Random File with at least 95% valid pixels")
#     plt.show()
# else:
#     print("No files with at least 99% valid pixels found.")

#################################################################################################################################
# Data augmentation
#################################################################################################################################

import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class RandomAdjustSharpness:
    def __init__(self, sharpness_range=(0.0, 1.0)):
        self.sharpness_range = sharpness_range

    def __call__(self, img):
        sharpness_factor = random.uniform(*self.sharpness_range)
        # Calculate the mean of the tensor as the gray value for each channel
        gray_value = img.mean(dim=(1, 2), keepdim=True)
        # Adjust sharpness for each channel separately
        img_sharp = (1 - sharpness_factor) * gray_value + sharpness_factor * img
        return img_sharp

class RandomColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        # Apply brightness
        img = img + torch.rand_like(img) * self.brightness
        # Apply contrast
        img_mean = img.mean(dim=(1, 2), keepdim=True)
        img = (img - img_mean) * (1 + torch.rand_like(img) * self.contrast) + img_mean
        # Apply saturation
        img_mean = img.mean(dim=(1, 2), keepdim=True)
        img_saturation = img - img_mean
        img = img_mean + img_saturation * (1 + torch.rand_like(img_saturation) * self.saturation)
        # Apply hue
        num_channels = img.size(0)
        for i in range(min(3, num_channels)):  # Apply hue to the first 3 channels (excluding the 4th channel if present)
            img[i] += self.hue
        return img

# Augment the data using PyTorch transformations
transform = transforms.Compose([
    #transforms.ToPILImage(),  # Convert tensor to PIL Image
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-45, 45)),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Use the custom RandomColorJitter
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    RandomAdjustSharpness(sharpness_range=(0.0, 1.0)),  # Use the custom RandomAdjustSharpness
    #transforms.ToTensor(), # Convert back to tensor
])

# Augment the data in batches
batch_size = 32
num_patches = len(train_input_patches)
num_batches = (num_patches + batch_size - 1) // batch_size
augmented_train_input_batches = []

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, num_patches)
    batch_images = train_input_patches[start_idx:end_idx]
    
    batch_augmented_images = torch.stack([transform(image) for image in batch_images])
    augmented_train_input_batches.append(batch_augmented_images)

# Concatenate the batches
augmented_train_input = torch.cat(augmented_train_input_batches)

# You can also augment your validation and test data:
# augmented_val_input = torch.stack([transform(image) for image in val_input_patches])
# augmented_test_input = torch.stack([transform(image) for image in test_input_patches])

# Visualize a sample from each tensor
def visualize_sample(tensor, title):
    # Convert tensor to numpy array and extract non-NaN data
    tensor_np = tensor[1000].cpu().numpy()
    valid_indices = ~np.isnan(tensor_np)
    valid_data = tensor_np[valid_indices]

    if len(valid_data) == 0:
        print(f"No valid data to visualize for '{title}'.")
        return

    plt.figure(figsize=(10, 5))
    plt.imshow(tensor_np[0], cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()

visualize_sample(augmented_train_input, 'Normalized Validation Input')

#################################################################################################################################
# Data filtering
#################################################################################################################################

import torch.nn.functional as F
import torchvision.transforms as transforms

# Define the filtering parameters (Three types of filters are applied: Gaussian filter, median filter, and bilateral filter.)
gaussian_sigma = 1.0
median_size = 3
bilateral_sigma_spatial = 2.0
bilateral_sigma_range = 0.1

# Function to apply Gaussian blur to input tensor
def apply_gaussian_blur(images, sigma):
    # Apply Gaussian blur to each image in the input tensor
    gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=sigma)
    return torch.stack([gaussian_blur(image) for image in images])

# Function to apply median filter to input tensor
def apply_median_filter(images, size):
    # Apply median filter to each image in the input tensor
    return torch.stack([transforms.median_filter(image.unsqueeze(0), kernel_size=size) for image in images])

# Function to apply bilateral filter to input tensor
def apply_bilateral_filter(images, sigma_spatial, sigma_range):
    # Apply bilateral filter to each image in the input tensor
    return torch.stack([transforms.bilateral_filter(image.unsqueeze(0), kernel_size=5, sigma_spatial=sigma_spatial, sigma_range=sigma_range) for image in images])

# Check if the input data is not empty
if len(augmented_train_input) > 0 and not torch.isnan(augmented_train_input).any():
    # Apply smoothing filters to the augmented input images
    smoothed_train_input = apply_gaussian_blur(augmented_train_input, gaussian_sigma)
    # median_filtered_train_input = apply_median_filter(augmented_train_input, median_size)
    # bilateral_filtered_train_input = apply_bilateral_filter(augmented_train_input, bilateral_sigma_spatial, bilateral_sigma_range)

print(smoothed_train_input.shape)
# Save the preprocessed data
smoothed_train_input_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\smoothed_train_input.pth"
smoothed_train_input = smoothed_train_input[:, :, :, :].permute(0, 2, 3, 1).to(device)
torch.save(smoothed_train_input, smoothed_train_input_path)
print("Smoothed train input data saved successfully.")
print(smoothed_train_input.shape)

# Visualize a sample from each tensor
def visualize_sample(tensor, title):
    # Convert tensor to numpy array and extract non-NaN data
    tensor_np = tensor[1].cpu().numpy()
    valid_indices = ~np.isnan(tensor_np)
    valid_data = tensor_np[valid_indices]

    if len(valid_data) == 0:
        print(f"No valid data to visualize for '{title}'.")
        return

    plt.figure(figsize=(10, 5))
    plt.imshow(tensor_np[0], cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()

visualize_sample(smoothed_train_input, 'Normalized Validation Input')

def normalize_data(data, chunk_size=100):
    """
    Normalize the input data to have values in the range [0, 1], considering only non-NaN data.

    :param data: numpy array or torch tensor containing the input data
    :param chunk_size: size of chunks to process
    :return: normalized data
    """
    num_samples = data.size(0)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size  # Calculate the number of chunks

    min_vals = []
    max_vals = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)

        chunk = data[start_idx:end_idx]
        valid_indices = ~torch.isnan(chunk)
        valid_data = torch.masked_select(chunk, valid_indices)

        if valid_data.numel() > 0:  # Check if there's valid data in the chunk
            min_vals.append(torch.min(valid_data))
            max_vals.append(torch.max(valid_data))

    if min_vals and max_vals:  # Check if valid values were found
        min_val = torch.min(torch.stack(min_vals))
        max_val = torch.max(torch.stack(max_vals))

        if min_val < 0 or max_val > 1:
            normalized_data = (data - min_val) / (max_val - min_val)
        else:
            normalized_data = data
    else:
        normalized_data = data

    return normalized_data

def prepare_validation_data(val_input, val_target, test_input, test_target): #Should not be neede, to check if results make sense
    """
    Prepare the validation and test data by normalizing and reshaping the input and target data.

    :param val_input: numpy array or torch tensor containing the validation input data
    :param val_target: numpy array or torch tensor containing the validation target data
    :param test_input: numpy array or torch tensor containing the test input data
    :param test_target: numpy array or torch tensor containing the test target data
    :return: normalized_val_input, normalized_val_target, normalized_test_input, normalized_test_target
    """
    # Normalize the input and target data
    normalized_val_input = normalize_data(val_input)
    normalized_val_target = normalize_data(val_target)
    normalized_test_input = normalize_data(test_input)
    normalized_test_target = normalize_data(test_target)

    # Transpose the input and target data to the shape expected by the model
    # Assuming val_input and val_target are of shape (samples, batch_size, height, width, channels) and Permute to (samples, channels, batch_size, height, width)
    normalized_val_input = normalized_val_input[:, :, :, :].permute(0, 2, 3, 1).to(device)
    #normalized_val_target = normalized_val_target[:, :, :].permute(0, 2, 3, 1).to(device)
    normalized_test_input = normalized_test_input[:, :, :, :].permute(0, 2, 3, 1).to(device)
    #normalized_test_target = normalized_test_target[:, :, :].permute(0, 2, 3, 1).to(device)

    return normalized_val_input, normalized_val_target, normalized_test_input, normalized_test_target

# Normalize and prepare the validation and test data
normalized_val_input, normalized_val_target, normalized_test_input, normalized_test_target = prepare_validation_data(val_input_patches, val_target_patches, test_input_patches, test_target_patches)

# Save the normalized validation and test data
normalized_val_input_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\normalized_val_input.pth"
normalized_val_target_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\normalized_val_target.pth"
normalized_test_input_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\normalized_test_input.pth"
normalized_test_target_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\normalized_test_target.pth"

torch.save(normalized_val_input, normalized_val_input_path)
torch.save(normalized_val_target, normalized_val_target_path)
torch.save(normalized_test_input, normalized_test_input_path)
torch.save(normalized_test_target, normalized_test_target_path)

print("Normalized validation and test data saved successfully.")

#################################################################################################################################
# Model Definition
#################################################################################################################################

# Load the preprocessed data
smoothed_train_input_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\smoothed_train_input.pth"
smoothed_train_input = torch.load(smoothed_train_input_path)

# Load the normalized validation and test data
normalized_val_input_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\normalized_val_input.pth"
normalized_val_target_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\normalized_val_target.pth"
normalized_test_input_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\normalized_test_input.pth"
normalized_test_target_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\normalized_test_target.pth"
train_target_patches_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\train_target_patches.pth"
normalized_val_input = torch.load(normalized_val_input_path)
normalized_val_target = torch.load(normalized_val_target_path)
normalized_test_input = torch.load(normalized_test_input_path)
normalized_test_target = torch.load(normalized_test_target_path)
train_target_patches = torch.load(normalized_test_target_path)

# Move data to the appropriate device
smoothed_train_input = smoothed_train_input.to(device)
normalized_val_input = normalized_val_input.to(device)
normalized_val_target = normalized_val_target.to(device)
normalized_test_input = normalized_test_input.to(device)
normalized_test_target = normalized_test_target.to(device)
train_target_patches = train_target_patches.to(device)

# !!! If you do not want to have any transformed data
smoothed_train_input = train_input_patches
smoothed_train_input = smoothed_train_input.permute(0, 2, 3, 1)
# !!!

# Visualize a sample from each tensor
def visualize_sample(tensor, title):
    # Convert tensor to numpy array and extract non-NaN data
    tensor_np = tensor[1].cpu().numpy()
    valid_indices = ~np.isnan(tensor_np)
    valid_data = tensor_np[valid_indices]

    if len(valid_data) == 0:
        print(f"No valid data to visualize for '{title}'.")
        return

    plt.figure(figsize=(10, 5))
    plt.imshow(tensor_np, cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()

visualize_sample(smoothed_train_input, 'Normalized Train Input')
visualize_sample(normalized_val_input, 'Normalized Validation Input')
visualize_sample(normalized_test_input, 'Normalized Test Input')
visualize_sample(normalized_val_target, 'Normalized Validation Target')
visualize_sample(normalized_test_target, 'Normalized Test Target')
visualize_sample(train_target_patches, 'Train Target Patches')

# Defining the different model architectures

def unet_model(input_shape, device):
    """
    Define the U-Net model architecture.

    :param input_shape: shape of the input tensor (excluding batch dimension)
    :param device: device to place the model on
    :return: a PyTorch Model representing the U-Net model
    """
    class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()

            # Custom encoder with 4 input channels
            self.encoder = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1).to(device),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1).to(device),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1).to(device),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1).to(device),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            # Decoder with upsampling layers to match spatial dimensions
            self.bridge = nn.Conv2d(512, 256, 3, padding=1).to(device)
            self.decoder = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1).to(device),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(128, 64, 3, padding=1).to(device),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(64, 1, 3, padding=1).to(device),  # Output 1 channel for regression (predicted height)
                # nn.Sigmoid()  # Remove Sigmoid for regression
            )

        def forward(self, x, batch_target):
            enc_features = self.encoder(x)
            bridge = self.bridge(enc_features)
            # Upsample the bridge features to match the spatial dimensions of the encoder features
            bridge = nn.functional.interpolate(bridge, size=enc_features.size()[2:], mode='bilinear', align_corners=True)
            # Concatenate the adjusted bridge features with the encoder features
            dec_features = torch.cat([enc_features, bridge], dim=1)
            # Add an additional convolutional layer to adjust the number of channels to 256
            dec_features = self.adjust_channels(dec_features)
            # Continue with the rest of the decoder
            dec_features = self.decoder(dec_features)
            # Upsample the output tensor to match the spatial dimensions of batch_target
            dec_features = F.interpolate(dec_features, size=batch_target.size()[2:], mode='bilinear', align_corners=True)
            return dec_features

        def adjust_channels(self, x):
            return self.adjust_conv(x)

        def adjust_conv(self, x):
            conv_layer = nn.Conv2d(x.shape[1], 256, kernel_size=1).to(device, dtype=torch.float32)
            return conv_layer(x)

    model = UNet().to(device)
    return model

def calculate_encoder_output_size(input_shape, device):
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # Custom encoder with 4 input channels
            self.encoder = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            # Calculate the output size of the encoder dynamically based on the input_shape
            with torch.no_grad():
                test_input = torch.zeros(1, *input_shape)  # Create a dummy tensor with batch size 1
                enc_features = self.encoder(test_input)
                self.encoder_output_size = enc_features.view(enc_features.size(0), -1).shape[1]

            # Classifier with fully connected layers
            self.classifier = nn.Sequential(
                nn.Linear(self.encoder_output_size, 128),  # Adjust the input size based on the encoder's output size
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),  # Output only 1 value for regression task (height prediction)
                # nn.Sigmoid()  # Remove Sigmoid for regression
            )

        def forward(self, x):
            enc_features = self.encoder(x)
            features = enc_features.view(enc_features.size(0), -1)
            output = self.classifier(features)
            return output

    model = CNN().to(device)
    return model

def resnet_model(input_shape, device):
    """
    Define the adapted ResNet model architecture for regression.

    :param input_shape: shape of the input tensor (excluding batch dimension)
    :return: a PyTorch Model representing the ResNet model for regression
    """

    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.features = models.resnet18(pretrained=True)
            self.features.conv1 = nn.Conv2d(
                input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False
            )  # Change the first layer to accept input_shape[0] channels
            self.features.fc = nn.Linear(512, 1)  # Output only 1 value for regression (height prediction)
            # No activation function (no nn.Sigmoid()) for regression

        def forward(self, x):
            features = self.features(x)
            output = features.view(features.size(0), -1)  # Flatten the output for regression
            return output

    model = ResNet().to(device)
    return model


def encoder_decoder_model(input_shape, device):
    """
    Define the adapted Encoder-Decoder model architecture for regression.

    :param input_shape: shape of the input tensor (excluding batch dimension)
    :return: a PyTorch Model representing the Encoder-Decoder model for regression
    """

    class EncoderDecoder(nn.Module):
        def __init__(self):
            super(EncoderDecoder, self).__init__()
            self.encoder = models.vgg16(pretrained=True).features[:23]
            self.encoder[0] = nn.Conv2d(
                input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False
            )  # Change the first layer to accept input_shape[0] channels
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.Conv2d(64, 1, 1),  # Output 1 channel for regression (predicted height)
                # No activation function (no nn.Sigmoid()) for regression
            )

        def forward(self, x):
            enc_features = self.encoder(x)
            dec_features = self.decoder(enc_features)
            return dec_features

    model = EncoderDecoder().to(device)
    return model

# Define the input shape of the U-Net model
input_shape = smoothed_train_input.shape[1:]

# Create the U-Net model
model = unet_model(input_shape, device)
# # Create the ResNet model
# model = resnet_model(input_shape, device)
# # Create the CNN model
#model = calculate_encoder_output_size(input_shape, device)
# # Create the Encoder-Decoder model
# model = encoder_decoder_model(input_shape, device)

# Move the model to the appropriate device
model = model.to(device)

# Check if the model is loaded onto the GPU(s)
if next(model.parameters()).is_cuda:
    print("Model is loaded on GPU.")
    # Checking if possible parallel computation
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
else:
    print("Model is not loaded on CPU.")

# Initialize the SummaryWriter object for logging (for tensorboard)
writer = SummaryWriter(log_dir="logs")

# Print the model summary
print(model)

# Save the model
model_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\U-net_model.pth"
torch.save(model.state_dict(), model_path)
print("Model saved successfully.")

#################################################################################################################################
# Model Training
#################################################################################################################################

# Define the batch size and number of epochs
batch_size = 16  # 32
epochs = 2 #10 #5000

# -------------------------------------------------------------------------------------------------------------
# Loss Functions
# -------------------------------------------------------------------------------------------------------------
mse_loss = nn.MSELoss(reduction='mean')

def mse_loss_no_nan(output, target):
    """
    Custom loss function for Mean Squared Error (MSE) loss, excluding NaN values.

    :param output: torch tensor representing the model's output
    :param target: torch tensor representing the target values
    :return: loss value
    """

    # Mask NaN values in the target tensor
    target_mask = ~torch.isnan(target)

    # Calculate the mean squared error loss, excluding NaN values
    loss = mse_loss(output[target_mask], target[target_mask])

    return loss

def rmse_loss_no_nan(output, target):
    """
    Custom loss function for Root Mean Square Error (RMSE) loss, excluding NaN values.

    :param output: torch tensor representing the model's output
    :param target: torch tensor representing the target values
    :return: loss value
    """

    # Mask NaN values in the target tensor
    target_mask = ~torch.isnan(target)

    # Calculate the squared error
    squared_error = (output - target) ** 2

    # Calculate the mean squared error, excluding NaN values
    mse_loss = torch.mean(squared_error[target_mask])

    # Calculate the root mean squared error
    rmse_loss = torch.sqrt(mse_loss)

    return rmse_loss

def mbe_loss_no_nan(output, target): # to check ! I still have NaN.
    """
    Custom loss function for Mean Bias Error (MBE) loss, excluding NaN values.

    :param output: torch tensor representing the model's output
    :param target: torch tensor representing the target values
    :return: loss value
    """

    # Mask NaN values in the target tensor
    target_mask = ~torch.isnan(target)

    # Filter NaN values from output and target tensors
    filtered_output = output[target_mask]
    filtered_target = target[target_mask]

    # Calculate the bias (mean error)
    bias = torch.mean(filtered_output - filtered_target)

    # Calculate the mean bias error, excluding NaN values
    mbe_loss = torch.mean((filtered_output - filtered_target - bias))

    return mbe_loss

bce_loss = nn.BCELoss(reduction='mean')

def bce_loss_no_nan(output, target): # to check ! I still have NaN.
    """
    Custom loss function for Binary Cross-Entropy (BCE) loss, excluding NaN values.

    :param output: torch tensor representing the model's output
    :param target: torch tensor representing the target values
    :return: loss value
    """

    # Mask NaN values in the target tensor
    target_mask = ~torch.isnan(target)

    # Filter NaN values from output and target tensors
    filtered_output = output[target_mask]
    filtered_target = target[target_mask]

    # Calculate the binary cross entropy loss, excluding NaN values
    loss = bce_loss(filtered_output, filtered_target)

    # Handle NaN values in the calculated loss
    valid_loss_values = loss[~torch.isnan(loss)]
    if valid_loss_values.numel() == 0:
        return torch.tensor(0.0, device=target.device, dtype=target.dtype)  # Return 0 loss if all loss values are NaN

    return valid_loss_values

def peak_signal_noise_ratio(y_true, y_pred):
    """
    Calculate the peak signal-to-noise ratio (PSNR) between the true and predicted values.

    :param y_true: true values
    :param y_pred: predicted values
    :return: peak signal-to-noise ratio
    """
    # Create a mask to ignore NaN values in both y_true and y_pred
    mask = ~torch.isnan(y_true) & ~torch.isnan(y_pred)

    # Apply the mask to y_true and y_pred to remove NaN values
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # Check if there are any NaN or Inf values
    if torch.any(torch.isnan(y_true_filtered)) or torch.any(torch.isnan(y_pred_filtered)):
        raise ValueError("Input tensors contain NaN or Inf values.")

    # Calculate the mean squared error (MSE) between the filtered y_true and y_pred
    mse = torch.mean((y_true_filtered - y_pred_filtered) ** 2)

    # Calculate the peak signal-to-noise ratio (PSNR)
    psnr = -10 * torch.log10(mse)

    return psnr.item()

# Training for Unet
#-------------------

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the current time
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/gradient_tape/" + current_time + "/train"
test_log_dir = "logs/gradient_tape/" + current_time + "/test"

# Start time measurement for training
start_time = time.time()

# Train the model (we cannot run the model on a CPU: DefaultCPUAllocator: not enough memory: you tried to allocate 99119792128 bytes.)
# Note: For the nn.MaxPool2d() function , it will raise the bug if kernel_size is bigger than its input_size.

history = {"loss": [], "val_loss": []}

# Ensure train_target_patches has the correct shape (4256, 1, 128, 128)
train_target_patches = train_target_patches.unsqueeze(1).to(device)

# # Determine the minimum number of elements in input_batches and target_batches
# num_elements = min(len(smoothed_train_input), len(train_target_patches))

# # Split the input and target data into batches
# input_batches = smoothed_train_input[:num_elements].split(batch_size)
# target_batches = train_target_patches[:num_elements].split(batch_size)

# # Split the input and target data into batches
input_batches = smoothed_train_input.split(batch_size)
target_batches = train_target_patches.split(batch_size)

# Define gradient accumulation steps
gradient_accumulation_steps = 4 #2

# Define a batch size for validation
val_batch_size = 8

# Define the shape of normalized_val_input
normalized_val_input = normalized_val_input.permute(0, 3, 1, 2)

# Lists to store evaluation metrics
rmse_values = []
psnr_values = []
mbe_values = []
bce_values = []


################################################################################################################################

# Define the size of the sub-sample
subsample_size = 30

for epoch in range(epochs):
    model.train()
    # Create a random sub-sample of indices
    random_indices = random.sample(range(len(input_batches)), subsample_size)
    for batch_idx in tqdm(random_indices, desc=f"Epoch {epoch + 1}/{epochs}", ncols=80):
        optimizer.zero_grad()

        # Process the batch_input and batch_target tensors per batch
        batch_input = input_batches[batch_idx].clone().detach().to(device, dtype=torch.float32)
        batch_target = target_batches[batch_idx].clone().detach().to(device, dtype=torch.float32)

        # Transpose the input tensors to (batch_size, channels, height, width)
        batch_input = batch_input.permute(0, 3, 1, 2)

        # Forward pass
        output = model(batch_input, batch_target)

        # Calculate the loss for this batch
        loss = mse_loss_no_nan(output, batch_target)
        total_loss = loss

        # Accumulate the batch loss to the total loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    # Calculate the average loss for the epoch
    avg_epoch_loss = total_loss / len(input_batches)

    # Print the average loss for the epoch
    print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}")

    # Display predictions on one random image from the batch
    model.eval()
    with torch.no_grad():
        random_image_idx = random.choice(random_indices)
        random_input = input_batches[random_image_idx].to(device, dtype=torch.float32)
        random_target = target_batches[random_image_idx].to(device, dtype=torch.float32)

        # Select one random image from the batch
        random_image = random_input[random.randint(0, random_input.shape[0] - 1)]

        predicted_output = model(random_image.unsqueeze(0).permute(0,3,1,2), random_target)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(random_image.cpu().numpy())
        plt.title("Input Image")

        plt.subplot(1, 3, 2)
        plt.imshow(random_target[0][0].cpu().numpy())
        plt.title("Target Image")

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_output[0][0].cpu().numpy())
        plt.title("Predicted Output")

        plt.show()

################################################################################################################################


for epoch in range(epochs):

    model.train()  # Set the model in training mode

    total_loss = 0.0  # Initialize total loss for accumulation

    for batch_idx in range(len(input_batches)):
        # Data preprocessing and preparation

        # Move the data to the GPU
        batch_input = input_batches[batch_idx].clone().detach().to(device, dtype=torch.float32)
        batch_target = target_batches[batch_idx].clone().detach().to(device, dtype=torch.float32)

        # Transpose the input tensors to (batch_size, channels, height, width)
        batch_input = batch_input.permute(0, 3, 1, 2)
        #batch_input = batch_input.permute(0, 2, 1, 3)

        # Forward pass
        output = model(batch_input, batch_target)
        loss = mse_loss_no_nan(output, batch_target)

        # Create a binary mask for valid values (1 for non-NaN, 0 for NaN)
        mask = ~torch.isnan(batch_target)

        # Apply the mask to the loss
        masked_loss = loss * mask

        # Compute the mean of masked_loss over valid elements
        masked_loss_mean = torch.sum(masked_loss) / torch.sum(mask)

        # Accumulate gradients
        masked_loss_mean = masked_loss_mean / gradient_accumulation_steps
        masked_loss_mean.backward()

        total_loss += masked_loss_mean.item()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients to prevent large values
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Perform optimization step
            optimizer.step()
            optimizer.zero_grad()

            # Empty GPU cache to release memory
            torch.cuda.empty_cache()

            print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx + 1}/{len(input_batches)}] Loss: {total_loss:.4f}")
            total_loss = 0.0

    # Validation at the end of the epoch
    model.eval()  # Set the model in evaluation mode
    val_loss_sum = 0.0

    with torch.no_grad():
        for val_batch_idx in range(0, len(normalized_val_input), val_batch_size):
            val_batch_input = normalized_val_input[val_batch_idx:val_batch_idx + val_batch_size].to(device)
            val_batch_target = normalized_val_target[val_batch_idx:val_batch_idx + val_batch_size].unsqueeze(1).to(device)

            val_output = model(val_batch_input, val_batch_target)
            val_loss = mse_loss_no_nan(val_output, val_batch_target)

            # Create a binary mask for valid values (1 for non-NaN, 0 for NaN)
            mask = ~torch.isnan(val_batch_target)
            masked_val_loss = val_loss * mask

            # Compute the mean of masked_val_loss over valid elements
            masked_val_loss_mean = torch.sum(masked_val_loss) / torch.sum(mask)

            val_loss_sum += masked_val_loss_mean.item()

            # Calculate additional evaluation metrics
            rmse = rmse_loss_no_nan(val_output, val_batch_target)
            psnr = peak_signal_noise_ratio(val_output, val_batch_target)
            mbe = mbe_loss_no_nan(val_output, val_batch_target)
            # bce = bce_loss_no_nan(val_output, val_batch_target)

            # Append metrics to lists
            rmse_values.append(rmse.item())
            psnr_values.append(psnr)
            mbe_values.append(mbe.item())
            # bce_values.append(bce.item())

        # Average validation loss over batches
        avg_val_loss = val_loss_sum / (len(normalized_val_input) // val_batch_size)

        print(f"Epoch [{epoch + 1}/{epochs}] Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation RMSE: {np.mean(rmse_values):.4f}")
        print(f"Validation PSNR: {np.mean(psnr_values):.4f}")
        print(f"Validation MBE: {np.mean(mbe_values):.4f}")
        # print(f"Validation BCE: {np.mean(bce_values):.4f}")

        # Clear the lists for the next epoch
        rmse_values.clear()
        psnr_values.clear()
        mbe_values.clear()
        # bce_values.clear()

    # Logging
    print(
        f"Epoch [{epoch+1}/{epochs}], Training Loss: {total_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
    )
    history["loss"].append(total_loss)
    history["val_loss"].append(avg_val_loss)

    # Logging (for tensorboard)
    writer.add_scalar("Training Loss", total_loss, epoch)
    writer.add_scalar("Validation Loss", avg_val_loss, epoch)

# End time measurement for training
end_time = time.time()
training_time = end_time - start_time
print("Training Time:", training_time)

# Close SummaryWriter when training is finished
writer.close()

# Save the trained model's state dictionary
#model_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\Trained_U-net_5000_epochs_batch_size_3.pth"
model_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\Trained_U-net_10_epochs_batch_size_16 - 11 August.pth"
torch.save(model.state_dict(), model_path)
print("Model saved successfully.")

#################################################################################################################################
# Model Evaluation Visualization
#################################################################################################################################

# -------------------------------------------------------------------------------------------------------------
# Visualization Functions
# -------------------------------------------------------------------------------------------------------------

def visualize_scalars(scalar_values, scalar_names, epochs=epochs):
    """
    Visualize scalar values over epochs.

    :param scalar_values: list of scalar values over time
    :param scalar_names: list of names corresponding to the scalar values
    :param epochs: number of epochs (default: 10)
    """
    epochs = list(range(1, epochs + 1))
    for i, values in enumerate(scalar_values):
        values = np.array(values)
        if values.ndim > 1:
            values = values.squeeze()
        plt.plot(epochs, values, label=scalar_names[i])
    plt.xlabel("Epochs")
    plt.ylabel("Scalar Values")
    plt.legend()
    plt.show()

def visualize_violin_plots(tensor_values, tensor_names):
    """
    Visualize violin plots of tensor distributions over time.

    :param tensor_values: list of tensor values over time
    :param tensor_names: list of names corresponding to the tensor values
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    for i, values in enumerate(tensor_values):
        sns.violinplot(data=values, inner="quart", label=tensor_names[i])

    plt.xlabel("Tensor Names")
    plt.ylabel("Value")
    plt.title("Violin Plots of Tensor Distributions")
    plt.legend()
    plt.show()

def visualize_histograms(tensor_values, tensor_names):
    """
    Visualize histograms of tensor values.

    :param tensor_values: list of tensor values
    :param tensor_names: list of names corresponding to the tensor values
    """
    for i, values in enumerate(tensor_values):
        flattened_values = np.concatenate(values)
        plt.hist(flattened_values, label=tensor_names[i], bins="auto")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def visualize_distributions(tensor_values, tensor_names):
    """
    Visualize distributions of tensor values over time.

    :param tensor_values: list of tensor values over time
    :param tensor_names: list of names corresponding to the tensor values
    """
    for i, values in enumerate(tensor_values):
        flattened_values = np.concatenate(values)
        plt.plot(flattened_values, label=tensor_names[i])
    plt.xlabel("Time")
    plt.ylabel("Tensor Value")
    plt.legend()
    plt.show()

# -------------------------------------------------------------------------------------------------------------
# Evaluating the Ouput
# -------------------------------------------------------------------------------------------------------------

# Calculate additional evaluation metrics after the loop
print(f"Final Validation Loss: {avg_val_loss:.4f}")
print(f"Root Mean Squared Error (RMSE): {np.mean(rmse_values):.4f}") #NaN
print(f"Peak Signal-to-Noise Ratio (PSNR): {np.mean(psnr_values):.4f}") #NaN
print(f"Mean Bias Error (MBE): {np.mean(mbe_values):.4f}") #NaN
#print("Binary Cross-Entropy (BCE):", bce)

# Visualize evaluation metrics after the loop
scalar_values = [rmse_values, psnr_values, mbe_values]
scalar_names = ["RMSE", "PSNR", "MBE"]
visualize_scalars(scalar_values, scalar_names)

# Visualize train_loss values over time
train_loss = [history["loss"]]
scalar_values = [train_loss]
scalar_names = ["Training Loss"]
visualize_scalars(scalar_values, scalar_names, epochs=11) # careful, the training has an extra epochs compared to the validation!

# Visualize val_loss values over time
val_loss = [history["val_loss"]]
scalar_values = [val_loss]
scalar_names = ["Validation Loss"]
visualize_scalars(scalar_values, scalar_names)

# Visualize scalar values (e.g., loss) over time
train_loss = [history["loss"]]
val_loss = [history["val_loss"]]
scalar_values = [train_loss, val_loss]
scalar_names = ["Training Loss", "Validation Loss"]
visualize_histograms(scalar_values, scalar_names)
visualize_violin_plots(scalar_values, scalar_names)

# Visualize histograms of tensor values (e.g., weights, biases)
# Convert the weights and biases from PyTorch tensors to NumPy arrays before plotting
weight_histograms = [
    layer.weight.detach().cpu().numpy().flatten()
    for layer in model.modules()
    if isinstance(layer, nn.Conv2d)
]
bias_histograms = [
    layer.bias.detach().cpu().numpy().flatten()
    for layer in model.modules()
    if isinstance(layer, nn.Conv2d)
]
tensor_values = [weight_histograms, bias_histograms]
tensor_names = ["Weight Histograms", "Bias Histograms"]
visualize_histograms(tensor_values, tensor_names)
visualize_violin_plots(tensor_values, tensor_names)

# Visualize distributions of tensor values over time
weight_distributions = [
    layer.weight.detach().cpu().numpy().flatten()
    for layer in model.modules()
    if isinstance(layer, nn.Conv2d)
]
bias_distributions = [
    layer.bias.detach().cpu().numpy().flatten()
    for layer in model.modules()
    if isinstance(layer, nn.Conv2d)
]
tensor_values = [weight_distributions, bias_distributions]
tensor_names = ["Weight Distributions", "Bias Distributions"]
visualize_distributions(tensor_values, tensor_names)

weight_tensor_values = [weight_distributions]
bias_tensor_values = [bias_distributions]
weight_tensor_names = ["Weight Distributions"]
bias_tensor_names = ["Bias Distributions"]

visualize_distributions(weight_tensor_values, weight_tensor_names)
visualize_distributions(bias_tensor_values, bias_tensor_names)

#################################################################################################################################
# Individual Model Prediction
#################################################################################################################################

# -------------------------------------------------------------------------------------------------------------
# Prediction at the patch level
# -------------------------------------------------------------------------------------------------------------

# Load the saved state dictionary into the model
model_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\Trained_U-net_10_epochs_batch_size_16 - 11 August.pth"
model.load_state_dict(torch.load(model_path))

# Put the model in evaluation mode
model.eval()

# Adjusting the shapes
normalized_test_input = normalized_test_input.permute(0, 3, 1, 2)
normalized_test_target = normalized_test_target.unsqueeze(1)

# Select a sample from the test set for prediction (takes optical images and releases a canopy height mask)
sample_index = 900 # A sample from the Test set is selected for prediction.
sample_input = normalized_test_input[sample_index]
sample_target = normalized_test_target[sample_index]

# Check for NaN values in the sample_input tensor
if torch.isnan(sample_target).any():
    print("Skipping prediction due to NaN values in the target.")
else:
    # Reshape the sample for prediction
    sample_input = torch.unsqueeze(torch.tensor(sample_input), dim=0).to(device, dtype=torch.float32)
    sample_target = torch.unsqueeze(torch.tensor(sample_target), dim=0).to(device, dtype=torch.float32)

    # Perform prediction
    with torch.no_grad():
        predicted_output = model(sample_input, sample_target)

# # Reshape the sample for prediction
# sample_input = torch.unsqueeze(torch.tensor(sample_input), dim=0).to(device, dtype=torch.float32)
# sample_target = torch.unsqueeze(torch.tensor(sample_target), dim=0).to(device, dtype=torch.float32)

# # Perform prediction
# with torch.no_grad():
#     predicted_output = model(sample_input, sample_target)

# Display the original, target, and predicted images
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

# Display each channel of the sample_input tensor as a separate subplot
for i in range(sample_input.shape[0]):
    axes[i].imshow(sample_input[i, 0].cpu(), cmap="viridis")
    axes[i].set_title(f"Input Channel {i}")
    axes[i].axis("off")

axes[sample_input.shape[0]].imshow(sample_target[0, 0, :, :].cpu(), cmap="viridis")
axes[sample_input.shape[0]].set_title("Target")
axes[sample_input.shape[0]].axis("off")

axes[sample_input.shape[0] + 1].imshow(predicted_output[0, 0, :, :].cpu(), cmap="viridis")
axes[sample_input.shape[0] + 1].set_title("Predicted")
axes[sample_input.shape[0] + 1].axis("off")

# Hide any remaining empty subplots
for i in range(sample_input.shape[0] + 2, 3):  # Use 3 as the upper limit for the loop
    axes[i].axis("off")

plt.show()

# Create a confusion plot with 2D histograms for ground truth and predictions # NOT WORKING: TO CHECK!
bin_size = 1
valid_samples_mask = (sample_target.shape[2] == 256) & (sample_target.shape[3] == 256)

# Select valid samples for plotting
valid_sample_indices = np.where(valid_samples_mask)[0]
sample_targets_valid = sample_target[valid_sample_indices]
predicted_outputs_valid = predicted_output[valid_sample_indices]

for sample_index in valid_sample_indices:
    sample_target = sample_targets_valid[sample_index]
    sample_input = normalized_test_input[sample_index]

    # Unsqueeze the tensors to account for the batch dimension
    sample_target = torch.unsqueeze(sample_target, dim=0)
    sample_input = torch.unsqueeze(sample_input, dim=0)

    hist_ground_truth, x_edges, y_edges = np.histogram2d(
        sample_target[0, 0, :, :].cpu().flatten(), sample_input[0, 0, :, :].cpu().flatten(),
        bins=[range(0, 100, bin_size), range(0, 100, bin_size)]
    )
    hist_predictions, _, _ = np.histogram2d(
        predicted_outputs_valid[sample_index, 0, :, :].cpu().numpy().flatten(), sample_input[0, 0, :, :].cpu().flatten(),
        bins=[x_edges, y_edges]
    )
    hist_ground_truth = hist_ground_truth.T
    hist_predictions = hist_predictions.T

    # Display the confusion plot for each valid sample
    plt.figure(figsize=(10, 5))
    plt.imshow(hist_ground_truth, origin='lower', extent=[0, 100, 0, 100], cmap='Blues')
    plt.colorbar(label='Count')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title(f'Confusion Plot - Ground Truth vs. Prediction (Sample {sample_index})')
    plt.tight_layout()
    plt.show()

# Calculate and display metrics between target and predicted data
mse_sample = mse_loss_no_nan(sample_target, predicted_output)
psnr_sample = peak_signal_noise_ratio(sample_target, predicted_output)
rmse_sample = rmse_loss_no_nan(sample_target, predicted_output)
mbe_sample = mbe_loss_no_nan(sample_target, predicted_output)
#bce_sample = bce_loss_no_nan(sample_target, predicted_output)

print("Sample Root Mean Squared Error (RMSE):", rmse_sample)
print("Sample Mean Bias Error (MBE):", mbe_sample)
#print("Sample Binary Cross-Entropy (BCE):", bce_sample)
print("Sample Mean Squared Error (MSE):", mse_sample)
print("Sample Peak Signal-to-Noise Ratio (PSNR):", psnr_sample)

# Create a single figure with two subplots for MSE and PSNR comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
colors = ['royalblue', 'limegreen']
# Plot MSE
axes[0].bar([0, 1], [mse_sample.cpu(), mse_loss_no_nan], tick_label=["Sample", "Overall"], alpha=0.8, color=colors)
axes[0].set_title('Mean Squared Error (MSE)')
axes[0].set_ylabel('Values')
# Plot PSNR
axes[1].bar([0, 1], [psnr_sample.cpu(), peak_signal_noise_ratio], tick_label=["Sample", "Overall"], alpha=0.8, color=colors)
axes[1].set_title('Peak Signal-to-Noise Ratio (PSNR)')
axes[1].set_ylabel('Values')
plt.tight_layout()
plt.show()

# Visualize the difference between target and predicted data
diff = np.abs(sample_target.cpu() - predicted_output.cpu().numpy())

# Calculate the range of the absolute difference based on non-NaN values
non_nan_diff = diff[~np.isnan(diff)]
non_nan_diff = np.nan_to_num(non_nan_diff)  # Convert NaN values to zeros

# Calculate the range of the non-NaN absolute difference
diff_range = np.ptp(non_nan_diff)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

# Select the first (and only) channel from the diff array before visualization
axes[0].imshow(diff[0, 0], cmap="viridis", vmin=0, vmax=diff_range)
axes[0].set_title("Absolute Difference")
axes[0].axis("off")

axes[1].imshow(sample_target[0, 0, :, :].cpu(), cmap="viridis")
axes[1].set_title("Target")
axes[1].axis("off")

axes[2].imshow(predicted_output[0, 0, :, :].cpu(), cmap="viridis")
axes[2].set_title("Predicted")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# Creating the cumulative distribution plot
predicted_heights = predicted_output.cpu().numpy().flatten()
sorted_heights = np.sort(predicted_heights)
cumulative_distribution = np.arange(1, len(sorted_heights) + 1) / len(sorted_heights)
plt.figure(figsize=(8, 6))
plt.plot(sorted_heights, cumulative_distribution, marker='o', linestyle='-', color='b')
plt.xlabel('Canopy Height Predictions (meters)')
plt.ylabel('Cumulative Distribution')
plt.title('Cumulative Distribution of Canopy Height Predictions')
plt.grid(True)
plt.show()

# Creating the precision-recall curve
prediction_errors = predicted_output - sample_target
prediction_errors = prediction_errors.squeeze()  # Squeeze the tensor to remove singleton dimensions
prediction_uncertainties = torch.std(prediction_errors, dim=(0, 1))  # Calculate along dimensions 1 (height) and 2 (width)
precision, recall = [], []
num_total = len(prediction_errors)
for i in range(1, num_total + 1):
    subset_errors = prediction_errors[:i]
    # Remove NaN values from the subset_errors tensor
    subset_errors_non_nan = subset_errors[~torch.isnan(subset_errors)]
    # Calculate the mean squared error for non-NaN values
    mse_non_nan = torch.mean(subset_errors_non_nan ** 2)
    # Calculate precision (RMSE) using the square root of the mean squared error
    precision.append(torch.sqrt(mse_non_nan).item())
    recall.append(i / num_total)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision (RMSE)")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# -------------------------------------------------------------------------------------------------------------
# Prediction at the tile level using the stacking of the patches to recreate the tile
# -------------------------------------------------------------------------------------------------------------

# TO ADJUST!!

# Specify the target size for the subtiles
target_shape = (256, 256)

# Step 1: Collect Sub-Tiles
subtile_dir = "C:/Users/mpetel/Documents/Kalimatan Project/Code/Data/Output/planet_tiles/Processed Planet/input_patches_within_95"
subtile_filenames = os.listdir(subtile_dir)

# Initialize a dictionary to store sub-tiles for each tile
tile_subtiles = {}

for filename in subtile_filenames:
    # Extract tile identification from filename (e.g., "001_utm_50S")
    tile_identification = "_".join(filename.split('_')[1:4])

    if tile_identification not in tile_subtiles:
        tile_subtiles[tile_identification] = []
    
    subtile_path = os.path.join(subtile_dir, filename)
    subtile_data = np.load(subtile_path)
    tile_subtiles[tile_identification].append(subtile_data)

# Step 2: Combine Sub-Tiles into Tiles and Predict on Tiles using Chunks
tile_tensors = {}
predicted_tile_outputs = {}

for tile_identification, subtiles in tile_subtiles.items():
    matching_subtiles = [subtile for subtile in subtiles if subtile.shape == target_shape]
    
    if len(matching_subtiles) > 0:
        combined_tile_data = np.concatenate(matching_subtiles, axis=0)
        tile_tensors[tile_identification] = torch.tensor(combined_tile_data).to(device, dtype=torch.float32)

# Step 3: Predict on Tiles using Chunks
chunk_size = 10  # Number of tiles to process in each chunk
all_tile_identifications = list(tile_subtiles.keys())

for i in range(0, len(all_tile_identifications), chunk_size):
    tile_chunk_identifications = all_tile_identifications[i:i + chunk_size]
    chunk_tensors = [tile_tensors[tile_identification] for tile_identification in tile_chunk_identifications]
    chunk_tensor = torch.stack(chunk_tensors)  # Stack chunk tensors to create a batch

    with torch.no_grad():
        predicted_outputs_chunk = model(chunk_tensor)

    for tile_identification, predicted_output in zip(tile_chunk_identifications, predicted_outputs_chunk):
        predicted_tile_outputs[tile_identification] = predicted_output

# Step 4: Visualization
for tile_identification, predicted_output in predicted_tile_outputs.items():
    # Load the ground truth LiDAR CHM data
    chm_filepath = f"C:/Users/mpetel/Documents/Kalimatan Project/Code/Data/Output/LiDAR/Processed LiDAR/{tile_identification}.tif"
    with rasterio.open(chm_filepath) as chm_data:
        ground_truth_chm = chm_data.read(1)

    # Visualize the ground truth CHM and predicted output side by side
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth_chm, cmap='viridis')
    plt.title("Ground Truth LiDAR CHM")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_output[0].squeeze(), cmap='viridis')
    plt.title("Predicted Output")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#################################################################################################################################
# Generate wall-to-wall tree height map
# Generalization of the individual prediction
#################################################################################################################################

# Perform prediction on the entire dataset (normalized_test_input) using the trained model
# Define the batch size for prediction
prediction_batch_size = 4

# Perform prediction on the entire dataset using the trained model
predicted_outputs = []
with torch.no_grad():
    for batch_start in range(0, len(normalized_test_input), prediction_batch_size):
        batch_input = normalized_test_input[batch_start:batch_start + prediction_batch_size].to(device)
        batch_target = normalized_test_target[batch_start:batch_start + prediction_batch_size].to(device)

        batch_predicted = model(batch_input, batch_target)
        predicted_outputs.append(batch_predicted)

        # Release GPU memory
        torch.cuda.empty_cache()

# Concatenate the predicted outputs into a single tensor
predicted_outputs = torch.cat(predicted_outputs, dim=0)

# Extract the tree height values from the predicted mask
tree_heights = predicted_outputs[:, :, :, :].cpu().numpy() # selecting first channel (index 0)

# Element-wise multiplication to create a wall-to-wall tree height map
tree_height_map = tree_heights * normalized_test_input.cpu().numpy()

# Convert the tree height map to a NumPy array with the desired data type
tree_height_map_np = tree_height_map.astype(np.float32)

# Save the tree height map to a new GeoTIFF file (Kalimantan_canopy_height.tif)
output_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\Kalimantan_canopy_height.tif"
height, width = tree_height_map.shape[2:]
bbox = (108, -5, 120, 7.5)
resolution = 1  # in meters
x_origin = bbox[0]
y_origin = bbox[3]
transform = rasterio.transform.from_origin(x_origin, y_origin, resolution, resolution)
crs = rasterio.crs.CRS.from_epsg(4326)

# Open the GeoTIFF file for writing
with rasterio.open(output_file, "w", driver="GTiff", height=height, width=width, count=4, dtype=tree_height_map_np.dtype,
                   crs=crs, transform=transform) as dst:
    for tile_idx in range(tree_height_map_np.shape[0]):  # Loop through the tiles
        for band_idx in range(tree_height_map_np.shape[1]):  # Loop through the bands
            band_data = tree_height_map_np[tile_idx, band_idx, :, :]
            dst.write(band_data, indexes=band_idx + 1)  # Write each band to the GeoTIFF

# Calculate the value range in meters and print it
value_min = tree_height_map.min()
value_max = tree_height_map.max()
print(f"Value range: {value_min} meters to {value_max} meters")

# Visualisations
#----------------

import datashader as ds
import datashader.transfer_functions as tf

# # NOT WORKING AT THE MOMENT: Visualize the generated wall-to-wall tree height map using Datashader
# tree_height_map_ds = ds.Canvas(plot_width=tree_height_map.shape[2], plot_height=tree_height_map.shape[3])
# agg = tree_height_map_ds.points(tree_height_map[0], agg=ds.by("band"))
# agg_shaded = tf.shade(agg, cmap=["blue", "green", "red"], how='linear')
# tf.set_background(agg_shaded, "black")
# tf.Image(agg_shaded).show()

# Histogram comparing the observed tree heights from the input data (ground truth) with the predicted tree heights from our model
observed_tree_heights = normalized_test_target[:, 0, :, :].cpu().numpy()  # Assuming the tree height is in the first channel
predicted_tree_heights = predicted_outputs[:, 0, :, :].cpu().numpy()
plt.hist(observed_tree_heights.flatten(), bins=50, alpha=0.5, color='blue', label='Observed')
plt.hist(predicted_tree_heights.flatten(), bins=50, alpha=0.5, color='orange', label='Predicted')
plt.xlabel('Tree Height')
plt.ylabel('Frequency')
plt.title('Distribution of Tree Heights: Observed vs. Predicted')
plt.legend()
plt.show()

# Scatter plot with observed tree heights on the x-axis and predicted tree heights on the y-axis.
from scipy.stats import linregress

# Flatten the observed and predicted tree heights arrays
observed_tree_heights_flat = observed_tree_heights.flatten()
predicted_tree_heights_flat = predicted_tree_heights.flatten()

# Find valid indices where both observed and predicted heights are not NaN
valid_indices = np.logical_and(~np.isnan(observed_tree_heights_flat), ~np.isnan(predicted_tree_heights_flat))

# Use valid indices to filter the data
observed_tree_heights_valid = observed_tree_heights_flat[valid_indices]
predicted_tree_heights_valid = predicted_tree_heights_flat[valid_indices]

# Calculate R-squared using vectorized operations
observed_mean = np.mean(observed_tree_heights_valid)
predicted_mean = np.mean(predicted_tree_heights_valid)
numerator = np.sum((observed_tree_heights_valid - observed_mean) * (predicted_tree_heights_valid - predicted_mean))
denominator_observed = np.sqrt(np.sum((observed_tree_heights_valid - observed_mean) ** 2))
denominator_predicted = np.sqrt(np.sum((predicted_tree_heights_valid - predicted_mean) ** 2))
r_squared = (numerator / (denominator_observed * denominator_predicted)) ** 2

# Calculate RMSE using vectorized operations
rmse = np.sqrt(np.mean((predicted_tree_heights_valid - observed_tree_heights_valid) ** 2))

# Create the scatter plot
plt.scatter(observed_tree_heights_valid, predicted_tree_heights_valid, alpha=0.5, color='blue', label='Data')
plt.plot(observed_tree_heights_valid, observed_tree_heights_valid, color='orange', label='Regression Line')
plt.xlabel('Observed Tree Heights')
plt.ylabel('Predicted Tree Heights')
plt.title(f'Observed vs. Predicted Tree Heights\nR2: {r_squared:.3f}, RMSE: {rmse:.3f}')
plt.legend()
plt.show()

# Heatmap for the first tile and first band
plt.figure(figsize=(10, 8))
sns.heatmap(tree_height_map[0, 0], cmap="viridis", annot=True, fmt=".1f")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Heatmap of Tree Heights for Tile 1, Band 1")
plt.show()

# Contour Plot
plt.figure(figsize=(10, 8))
plt.contour(tree_height_map[0, 0], cmap="viridis")
plt.colorbar(label="Tree Height (m)")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Contour Plot of Tree Heights")
plt.show()

# 3D Plot
from mpl_toolkits.mplot3d import Axes3D
x, y = np.meshgrid(np.arange(width), np.arange(height))
z = tree_height_map[0, 0]
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Tree Height (m)")
ax.set_title("3D Plot of Tree Heights")
plt.show()

# Kernel Density Estimation (KDE) Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(tree_height_map_np.flatten(), shade=True)
plt.xlabel("Tree Height (m)")
plt.ylabel("Density")
plt.title("Kernel Density Estimation (KDE) Plot of Tree Heights")
plt.show()

#################################################################################################################################
# Generate wall-to-wall variance map
#################################################################################################################################

# Calculate the variance of tree heights across the entire dataset
tree_height_var = np.var(tree_height_map, axis=0)

# Convert the tree height map to a NumPy array with the desired data type
tree_height_var_map_np = tree_height_var.cpu().numpy().astype(np.float32)

# Save the variance map to a new GeoTIFF file
output_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\Kalimantan_tree_height_variance.tif"
height, width = tree_height_var.shape[1:]
transform = rasterio.transform.from_origin(x_origin, y_origin, resolution, resolution)
crs = rasterio.crs.CRS.from_epsg(4326)

with rasterio.open(output_file, "w", driver="GTiff", height=height, width=width, count=4, dtype=tree_height_var_map_np.dtype,
                   crs=crs, transform=transform) as dst:
    dst.write(tree_height_var_map_np, 1)

# Calculate the value range in meters and print it
value_min = tree_height_var.min()
value_max = tree_height_var.max()
print(f"Value range: {value_min} meters to {value_max} meters")

# Visualisations
#----------------

# Visualize the generated variance map
plt.imshow(tree_height_var, cmap="coolwarm")
plt.colorbar(label="Tree Height Variance")
plt.title("Tree Height Variance Map")
plt.show()

# NOT WORKING AT THE MOMENT: Visualize the generated variance map using Datashader
variance_map_ds = ds.Canvas(plot_width=tree_height_var.shape[1], plot_height=tree_height_var.shape[0])
agg = variance_map_ds.tf.shade(tree_height_var, cmap="coolwarm", how='linear')
tf.set_background(agg, "black")
tf.Image(agg).show()

# Histogram comparing the observed tree heights from the input data (ground truth) with the predicted tree heights from our model

observed_tree_heights = normalized_test_target[:, 0, :, :].numpy()  # Assuming the tree height is in the first channel
predicted_tree_heights = predicted_outputs[:, 0, :, :].numpy()
plt.hist(observed_tree_heights.flatten(), bins=50, alpha=0.5, color='blue', label='Observed')
plt.hist(predicted_tree_heights.flatten(), bins=50, alpha=0.5, color='orange', label='Predicted')
plt.xlabel('Tree Height')
plt.ylabel('Frequency')
plt.title('Distribution of Tree Heights: Observed vs. Predicted')
plt.legend()
plt.show()

# Scatter plot with observed tree heights on the x-axis and predicted tree heights on the y-axis.
from scipy.stats import linregrError
observed_tree_heights_flat = observed_tree_heights.flatten()
predicted_tree_heights_flat = predicted_tree_heights.flatten()
valid_indices = ~np.isnan(observed_tree_heights_flat) & ~np.isnan(predicted_tree_heights_flat)
observed_tree_heights_valid = observed_tree_heights_flat[valid_indices]
predicted_tree_heights_valid = predicted_tree_heights_flat[valid_indices]
r_squared = np.corrcoef(observed_tree_heights_valid, predicted_tree_heights_valid)[0, 1] ** 2
rmse = np.sqrt(np.mean((predicted_tree_heights_valid - observed_tree_heights_valid) ** 2))
plt.scatter(observed_tree_heights_valid, predicted_tree_heights_valid, alpha=0.5, color='blue', label='Data')
plt.plot(observed_tree_heights_valid, observed_tree_heights_valid, color='orange', label='Regression Line')
plt.xlabel('Observed Tree Heights')
plt.ylabel('Predicted Tree Heights')
plt.title(f'Observed vs. Predicted Tree Heights\nR2: {r_squared:.3f}, RMSE: {rmse:.3f}')
plt.legend()
plt.show()

# Calculate confidence intervals for predicted tree heights
confidence_intervals = 1.96 * np.std(predicted_tree_heights, axis=0) / np.sqrt(len(predicted_tree_heights))
plt.errorbar(observed_tree_heights_flat, predicted_tree_heights_flat, yerr=confidence_intervals,
             fmt='o', markersize=4, alpha=0.5, color='blue', label='Data')
plt.plot(observed_tree_heights_flat, observed_tree_heights_flat, color='orange', label='Regression Line')
plt.xlabel('Observed Tree Heights')
plt.ylabel('Predicted Tree Heights')
plt.title(f'Observed vs. Predicted Tree Heights\nR2: {r_squared:.3f}, RMSE: {rmse:.3f}')
plt.legend()
plt.show()

# Create box plots to visualize the distribution of observed and predicted tree heights
plt.boxplot([observed_tree_heights_flat, predicted_tree_heights_flat], labels=['Observed', 'Predicted'])
plt.ylabel('Tree Heights')
plt.title('Box Plot of Observed and Predicted Tree Heights')
plt.show()

#################################################################################################################################
# Compute and save Kalimantan_percent_cover.tif
#################################################################################################################################

# Calculate percent canopy cover by counting pixels higher than 5 meters within each 1-ha cell
threshold = 5  # Threshold height in meters

# Create a mask to exclude NaN values from the computation
mask = ~np.isnan(tree_height_map_np)

# Calculate percent canopy cover by counting pixels higher than the threshold within each 1-ha cell
cell_size = 100  # Assuming 1-ha cell size in meters
percent_cover = np.zeros_like(tree_height_map_np)

for row in range(0, tree_height_map.shape[0], cell_size):
    for col in range(0, tree_height_map.shape[1], cell_size):
        cell = tree_height_map[row:row+cell_size, col:col+cell_size]
        valid_pixels = cell[mask[row:row+cell_size, col:col+cell_size]]
        percent_cover[row:row+cell_size, col:col+cell_size] = np.count_nonzero(valid_pixels > threshold) / valid_pixels.size * 100

# Save the percent canopy cover map to a new GeoTIFF file (Kalimantan_percent_cover.tif)
output_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\Kalimantan_percent_cover.tif"
height, width = percent_cover.shape
transform = rasterio.transform.from_origin(x_origin, y_origin, cell_size, cell_size)  # Assuming 1-ha grid with cell_size m resolution
crs = rasterio.crs.CRS.from_epsg(4326)

with rasterio.open(output_file, "w", driver="GTiff", height=height, width=width, count=1, dtype=percent_cover.dtype,
                   crs=crs, transform=transform) as dst:
    dst.write(percent_cover, 1)

# Visualize the generated percent canopy cover map using a suitable visualization library
plt.imshow(percent_cover, cmap="viridis")
plt.colorbar(label="Percent Canopy Cover (%)")
plt.title("Percent Canopy Cover Map")
plt.show()


#################################################################################################################################
# Compute and save Kalimantan_LCA.tif
#################################################################################################################################

# To compute the Large Crown Area (LCA) within each 1-ha grid cell using the LCA algorithm, we'll follow the steps described:
# Apply a threshold of 27 meters on the 1-m CHM to derive a binary image representing the areas covered by high vegetation (large trees).
# Perform connected component segmentation with 8-neighborhood pixel connectivity to compute clusters of pixels representing large tree areas.
# Remove clusters composed of less than 100 pixels (i.e., 100 m²) to get the LCA binary image at 1-m resolution.
# Resample the LCA binary image to a 1-ha grid to represent the percentage of area covered by large trees within each 1-ha cell.

from scipy import ndimage

# Threshold the 1-m CHM to get a binary image representing high vegetation areas (large trees)
lca_threshold = 27
binary_lca = tree_height_map > lca_threshold

# Perform connected component segmentation to compute clusters of large tree areas
lca_clusters, num_clusters = ndimage.label(binary_lca, structure=np.ones((3, 3)))

# Remove clusters smaller than 100 pixels (100 m²)
min_cluster_size = 100
for cluster_label in range(1, num_clusters + 1):
    cluster_size = np.sum(lca_clusters == cluster_label)
    if cluster_size < min_cluster_size:
        lca_clusters[lca_clusters == cluster_label] = 0

# Convert the large tree clusters to 1-ha grid by summing up the pixels within each 1-ha cell
lca_1ha_grid = np.zeros((tree_height_map.shape[0] // cell_size, tree_height_map.shape[1] // cell_size))
for row in range(0, tree_height_map.shape[0], cell_size):
    for col in range(0, tree_height_map.shape[1], cell_size):
        cell = lca_clusters[row:row+cell_size, col:col+cell_size]
        lca_1ha_grid[row // cell_size, col // cell_size] = np.sum(cell > 0)

# Compute the percentage of area covered by large trees for each 1-ha cell
lca_percent_cover = (lca_1ha_grid / (cell_size ** 2)) * 100

# Save the Large Crown Area (LCA) percent cover map to a new GeoTIFF file (Kalimantan_LCA.tif)
output_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\Kalimantan_LCA.tif"
height, width = lca_percent_cover.shape
transform_lca = rasterio.transform.from_origin(x_origin, y_origin, cell_size, cell_size)  # Assuming 1-ha grid with cell_size m resolution

with rasterio.open(output_file_lca, "w", driver="GTiff", height=height, width=width, count=1, dtype=lca_percent_cover.dtype,
                   crs=crs, transform=transform_lca) as dst:
    dst.write(lca_percent_cover, 1)

# Visualize the generated Large Crown Area (LCA) percent cover map using a suitable visualization library
plt.imshow(lca_percent_cover, cmap="viridis")
plt.colorbar(label="LCA Percent Cover (%)")
plt.title("Large Crown Area (LCA) Percent Cover Map")
plt.show()


#################################################################################################################################
# Compute and save Kalimantan_degradation_index.tif
#################################################################################################################################

# Calculate the Forest Degradation Index (FDI) as FDI = MCH + LCA + PC
# Assuming MCH (mean crown height) is the same as tree_height_map
fdi = tree_height_map + lca_percent_cover + percent_cover

# Save the Forest Degradation Index (FDI) map to a new GeoTIFF file (Kalimantan_degradation_index.tif)
output_file_fdi = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95%\Kalimantan_degradation_index.tif"

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

height, width = fdi_classes.shape

with rasterio.open(output_file_fdi, "w", driver="GTiff", height=height, width=width, count=1, dtype=fdi_classes.dtype,
                   crs=crs, transform=transform) as dst:
    dst.write(fdi_classes, 1)

# Visualize the generated Forest Degradation Index (FDI) map using a suitable visualization library
plt.imshow(fdi_classes, cmap="viridis")
plt.colorbar(label="Forest Degradation Index (FDI)")
plt.title("Forest Degradation Index (FDI) Map")
plt.show()

#################################################################################################################################
# ABG computation
#################################################################################################################################

def calculate_aboveground_biomass(d, wd, E=-0.1, coefficients=None):
    """
    Calculate tree-level aboveground biomass (AGB) based on the provided formula.

    :param d: Diameter of the bole measured at diameter at breast height or above the buttress (cm).
    :param wd: Wood density (g cm^-3) that depends on the tree species.
    :param E: Measure of environmental stress (default is -0.1).
    :param coefficients: Coefficients for the AGB formula (optional, default values used if None).
    :return: Tree-level aboveground biomass (AGB) calculated using the formula.
    """
    if wd is None:
        wd = 0.57  # Average value for Kalimantan

    if coefficients is None:
        coefficients = {
            'c0': -1.803,
            'c1': -0.976,
            'c2': 0.976,
            'c3': 2.673,
            'c4': -0.0299
        }
    
    c0 = coefficients['c0']
    c1 = coefficients['c1']
    c2 = coefficients['c2']
    c3 = coefficients['c3']
    c4 = coefficients['c4']

    AGB = np.exp(c0 + c1 * E + c2 * np.log(wd) + c3 * np.log(d) + c4 * (np.log(d))**2)
    return AGB

def calculate_aboveground_carbon_density(TCH, Cover_resid, BA, WD, rho):
    """
    Calculate aboveground carbon density (ACD) based on the given parameters and coefficients.

    :param TCH: Tree canopy height (m).
    :param Cover_resid: Residual canopy cover.
    :param BA: Basal area (m^2/ha).
    :param WD: Wood density (g cm^-3).
    :param rho: Coefficients (ρ0-3) as a list or array.
    :return: Aboveground carbon density (ACD) in units to be consistent with the provided coefficients.
    """
    rho0, rho1, rho2, rho3 = rho
    BA_estimated = rho0 * (TCH ** rho1) * (1 + rho2 * Cover_resid)
    WD_estimated = rho0 * (TCH ** rho1)
    ACD = BA_estimated ** rho2 * WD_estimated ** rho3
    return ACD

# Implementation of Monte Carlo simulations to estimate uncertainty:

# Define uncertain parameters and their distributions
num_simulations = 1000

# Generate random samples for the coefficients of AGB and ACD models
uncertain_agb_coeff_samples = np.random.normal(mean_agb_coeffs, std_agb_coeffs, (num_simulations, 5))
uncertain_acd_rho_samples = np.random.normal(mean_acd_rho, std_acd_rho, (num_simulations, 4))

# Placeholder for storing predicted values from simulations
predicted_agb_values = []
predicted_acd_values = []

# Run simulations for AGB
for agb_coeffs in uncertain_agb_coeff_samples:
    AGB_prediction = calculate_aboveground_biomass(d, wd, E, agb_coeffs)
    predicted_agb_values.append(AGB_prediction)

# Run simulations for ACD
for acd_rho_values in uncertain_acd_rho_samples:
    ACD_prediction = calculate_aboveground_carbon_density(TCH, Cover_resid, BA, WD, acd_rho_values)
    predicted_acd_values.append(ACD_prediction)

# Analyze results for AGB and ACD
mean_agb_values = np.mean(predicted_agb_values, axis=0)
std_agb_values = np.std(predicted_agb_values, axis=0)
mean_acd_values = np.mean(predicted_acd_values, axis=0)
std_acd_values = np.std(predicted_acd_values, axis=0)

print("Mean AGB Values:", mean_agb_values)
print("Standard Deviation of AGB Values:", std_agb_values)
print("Mean ACD Values:", mean_acd_values)
print("Standard Deviation of ACD Values:", std_acd_values)

def calculate_agb_plot_level(d_list, wd_list, area):
    """
    Calculate plot-level aboveground biomass (AGBp) in Mgha^-1.

    :param d_list: List of diameters of boles measured at diameter at breast height or above the buttress (cm).
    :param wd_list: List of wood densities (g cm^-3) that depend on the tree species.
    :param area: Area of the plot in hectares (ha).
    :return: Plot-level aboveground biomass (AGBp) in Mgha^-1.
    """
    AGB_tree_level = [calculate_agb_tree_level(d, wd) for d, wd in zip(d_list, wd_list)]
    AGB_plot_level = np.sum(AGB_tree_level) / (area * 0.0001)  # Convert area to m²
    return AGB_plot_level


# Example data for demonstration (replace with actual data)
diameters = [15, 20, 25]  # Diameter of boles measured at diameter at breast height or above the buttress (cm)
wood_densities = [0.6, 0.55, 0.65]  # Wood densities (g cm^-3) for each tree species
plot_area = 1  # Area of the plot in hectares (ha)

# Calculate plot-level aboveground biomass (AGBp) using the sub-sampling strategy
AGB_plot_level_sub_sampling = calculate_agb_plot_level(diameters, wood_densities, plot_area)

# Calculate plot-level aboveground biomass (AGBp) using the tree-level strategy
AGB_tree_level = [calculate_agb_tree_level(d, wd) for d, wd in zip(diameters, wood_densities)]
AGB_plot_level_tree_level = np.sum(AGB_tree_level) / (plot_area * 0.0001)  # Convert area to m²

# Save the plot-level AGB values in a GeoTIFF file (Kalimantan_aboveground_biomass.tif)
output_file_agb = "Kalimantan_aboveground_biomass.tif"
height, width = 1, 1
transform_agb = rasterio.transform.from_origin(0, 0, 1, 1)  # Assuming 1-ha grid with 1 m resolution
crs_agb = rasterio.crs.CRS.from_epsg(4326)  # Assuming WGS84 coordinate reference system

with rasterio.open(output_file_agb, "w", driver="GTiff", height=height, width=width, count=1, dtype='float32',
                   crs=crs_agb, transform=transform_agb) as dst:
    dst.write(AGB_plot_level_tree_level, 1)

# Print the range of values measured in Mg biomass/ha
print(f"Plot-level AGB using sub-sampling strategy: {AGB_plot_level_sub_sampling} Mgha^-1")
print(f"Plot-level AGB using tree-level strategy: {AGB_plot_level_tree_level} Mgha^-1")

#############################################################################################################################
#############################################################################################################################


























# #############################################################################################################################
# # Pre-processing functions - OLD
# #############################################################################################################################

# # def truncate_and_scale(image, truncate_min, truncate_max, scale_factor):
# #     image = np.clip(image, truncate_min, truncate_max)
# #     image = image * scale_factor
# #     return image


# # def scale_to_8bits(image):
# #     image = image / 10
# #     image = np.clip(image, 0, 255)
# #     image = image.astype(np.uint8)
# #     return image


# # def create_composite(bands):
# #     red_band = truncate_and_scale(bands["red"], 0, 2540, 1)
# #     green_band = truncate_and_scale(bands["green"], 0, 2540, 1)
# #     blue_band = truncate_and_scale(bands["blue"], 0, 2540, 1)
# #     nir_band = truncate_and_scale(bands["nir"], 0, 10000, 1 / 3.937)

# #     red_band = scale_to_8bits(red_band)
# #     green_band = scale_to_8bits(green_band)
# #     blue_band = scale_to_8bits(blue_band)
# #     nir_band = scale_to_8bits(nir_band)

# #     composite = cv2.merge([red_band, green_band, blue_band, nir_band])
# #     return composite


# # def add_border(image, border_size):
# #     image_with_border = cv2.copyMakeBorder(
# #         image, border_size, border_size, border_size, border_size, cv2.BORDER_REFLECT
# #     )
# #     return image_with_border


# # # Path to the folder containing the TIFF files
# # folder_path = r"C:\Users\Matéo\OneDrive\Documents\GAP YEAR 2022-2023\NASA JPL\JPL Project\Code\Data\output\optical"

# # # List all TIFF files in the folder
# # image_files = [
# #     os.path.join(folder_path, file)
# #     for file in os.listdir(folder_path)
# #     if file.endswith(".tiff")
# # ]

# # bands = {"red": [], "green": [], "blue": [], "nir": []}

# # # Read and collect the individual bands from TIFF files
# # for file in image_files:
# #     image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
# #     bands["red"].append(image[:, :, 0])
# #     bands["green"].append(image[:, :, 1])
# #     bands["blue"].append(image[:, :, 2])
# #     bands["nir"].append(image[:, :, 3])

# # # Convert the band lists to NumPy arrays
# # for key in bands.keys():
# #     bands[key] = np.array(bands[key])

# # # Create the RGBNIR composite image
# # composite_image = create_composite(bands)

# # # Add border to the composite image
# # border_size = 128
# # composite_image_with_border = add_border(composite_image, border_size)

# # # Save the composite image with border
# # output_path = os.path.join(folder_path, "composite_with_border.tiff")
# # cv2.imwrite(output_path, composite_image_with_border)
