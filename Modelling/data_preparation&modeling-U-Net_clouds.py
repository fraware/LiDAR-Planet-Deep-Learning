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
from dask.diagnostics import ProgressBar
import skimage.util as sk_util
import plotly.express as px
from skimage.transform import resize
import tifffile as tiff
import imgaug.augmenters as iaa
import glob
import torch
torch.cuda.empty_cache()
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_tensor
import datetime
import time
import shutil
import torch
import torch.nn as nn
import torch.functional as F
from torchviz import make_dot
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
input_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet U-Net"  # Optical
target_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR"  # LiDAR

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
    valid_pixels = np.count_nonzero(data <= 254)
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

def process_files_in_folder(planet_file_paths, chm_file_paths, chunk_size=(1000, 1000)):
    """
    Reproject and save the Planet files to match the projection of LiDAR files using chunk processing.

    Parameters:
        planet_file_paths (list): List of file paths to the Planet files.
        chm_file_paths (list): List of file paths to the LiDAR (CHM) files.
        chunk_size (tuple): Size of chunks for processing. Default is (1000, 1000).

    Returns:
        None
    """
    reprojected_files = []  # Store paths of reprojected files

    for planet_file_path, chm_file_path in zip(planet_file_paths, chm_file_paths):
        try:
            with xr.open_dataset(planet_file_path, chunks={'band_data': chunk_size}) as planet_data, \
                 xr.open_dataset(chm_file_path) as chm_data:

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

    # ! Delete the original files after all files have been reprojected, closed, and renamed
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
    input_files = sorted(glob.glob(os.path.join(input_folder, "input_patches", "*.npy")))
    target_files = sorted(glob.glob(os.path.join(target_folder, "target_patches", "*.npy")))

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

    # Remove unmatched input files
    unmatched_input_files = set(input_files) - set(pair[0] for pair in matching_pairs)
    for unmatched_input_file in unmatched_input_files:
        try:
            os.remove(unmatched_input_file)
        except Exception as e:
            print(f"Error removing unmatched input file {os.path.basename(unmatched_input_file)}: {str(e)}")

    # Remove unmatched target files
    unmatched_target_files = set(target_files) - set(pair[1] for pair in matching_pairs)
    for unmatched_target_file in unmatched_target_files:
        try:
            os.remove(unmatched_target_file)
        except Exception as e:
            print(f"Error removing unmatched target file {os.path.basename(unmatched_target_file)}: {str(e)}")

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
        rounded_input_patches_array, rounded_target_patches_array, test_size=0.2, random_state=42, shuffle=True
    )

    train_input, val_input, train_target, val_target = train_test_split(
        train_input, train_target, test_size=0.5, random_state=42, shuffle=True
    )

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
planet_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet U-Net\input_patches"
lidar_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\target_patches"
output_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95% - Un-Net"
load_and_preprocess_data(planet_folder_path, lidar_folder_path, output_folder)

# Converting the data to tensors
output_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95% - Un-Net"
train_input_path = os.path.join(output_folder, "train_input.npy")
val_input_path = os.path.join(output_folder, "val_input.npy")
test_input_path = os.path.join(output_folder, "test_input.npy")
train_target_path = os.path.join(output_folder, "train_target.npy")
val_target_path = os.path.join(output_folder, "val_target.npy")
test_target_path = os.path.join(output_folder, "test_target.npy")

train_input_patches, val_input_patches, test_input_patches, train_target_patches, val_target_patches, test_target_patches = \
    convert_to_tensors(train_input_path, val_input_path, test_input_path, train_target_path, val_target_path, test_target_path)

train_target_patches_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95% - Un-Net\train_target_patches.pth"
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

# folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\target_patches"
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
# tiff_file_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet U-Net\Polygon_002_utm_50N_merged_modified.tif"
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
# npy_file_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet U-Net\input_patches\Polygon_001_utm_50S_merged_modified_patch_768_4096.npy"
# check_number_of_bands(npy_file_path)


# def check_npy_shapes(folder_path):
#     npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

#     for npy_file in npy_files:
#         npy_path = os.path.join(folder_path, npy_file)
#         npy_array = np.load(npy_path)
#         print(f"Shape of {npy_file}: {npy_array.shape}")

# # Path to .npy files
# #folder_path = r'C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\target_patches'
# folder_path = r'C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet U-Net\input_patches'
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
# #folder_path = r'C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\target_patches'
# folder_path = r'C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet U-Net\input_patches'
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
# npy_file_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet U-Net\input_patches\Polygon_001_utm_50S_merged_modified_patch_768_4096.npy"
# normalize_single_input(npy_file_path)

# # -------------------------------------------------------------------------------------------------------------
# # Image Comparison Tests
# # -------------------------------------------------------------------------------------------------------------

# # Paths to npy files
# planet_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet U-Net\input_patches"
# lidar_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\target_patches"
# input_path = os.path.join(planet_folder_path, "Polygon_001_utm_50S_merged_modified_patch_768_4096.npy")
# target_path = os.path.join(lidar_folder_path, "Polygon_001_utm_50S_patch_768_4096.npy")

# # Get a list of all npy files in the folder
# input_npy_files = [f for f in os.listdir(planet_folder_path) if f.endswith(".npy")]
# target_npy_files = [f for f in os.listdir(lidar_folder_path) if f.endswith(".npy")]

# # Select a random input and target npy file
# random_input_file = random.choice(input_npy_files)
# random_target_file = random.choice(target_npy_files)

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
# planet_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet U-Net\input_patches"

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
smoothed_train_input_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95% - Un-Net\smoothed_train_input.pth"
smoothed_train_input = smoothed_train_input[:, :, :, :].permute(0, 2, 3, 1).to(device)
torch.save(smoothed_train_input, smoothed_train_input_path)
print("Smoothed train input data saved successfully.")
print(smoothed_train_input.shape)

train_input_patches_patch = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95% - Un-Net\train_input_patches.pth"
train_input_patches = train_input_patches[:, :, :, :].permute(0, 2, 3, 1).to(device)
train_input_patches = train_input_patches[:, :, :, :].permute(0, 2, 1, 3).to(device)
torch.save(train_input_patches, train_input_patches_patch)

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
normalized_val_input_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95% - Un-Net\normalized_val_input.pth"
normalized_val_target_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95% - Un-Net\normalized_val_target.pth"
normalized_test_input_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95% - Un-Net\normalized_test_input.pth"
normalized_test_target_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 95% - Un-Net\normalized_test_target.pth"

torch.save(normalized_val_input, normalized_val_input_path)
torch.save(normalized_val_target, normalized_val_target_path)
torch.save(normalized_test_input, normalized_test_input_path)
torch.save(normalized_test_target, normalized_test_target_path)

print("Normalized validation and test data saved successfully.")

