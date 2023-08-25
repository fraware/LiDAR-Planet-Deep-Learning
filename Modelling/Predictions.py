# The purpose of this file is to perform model prediction at different levels.

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
from Modelling import unet_model, calculate_encoder_output_size, resnet_model, encoder_decoder_model

#################################################################################################################################
# Part 2: Defining paths and parameters
#################################################################################################################################

# Define the paths to the input and target data folders
input_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet"  # Optical
target_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR"  # LiDAR

# Load the data
smoothed_train_input = smoothed_train_input.to(device)
normalized_val_input = normalized_val_input.to(device)
normalized_val_target = normalized_val_target.to(device)
normalized_test_input = normalized_test_input.to(device)
normalized_test_target = normalized_test_target.to(device)
train_target_patches = train_target_patches.to(device)

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

# Define the input shape of the U-Net model
input_shape = smoothed_train_input.shape[1:] #4
n_classes = 1

# Create the U-Net model
model = unet_model(input_shape, n_classes)
# Create the ResNet model
# model = resnet_model(input_shape, device)
# Create the CNN model
# model = calculate_encoder_output_size(input_shape, device)
# Create the Encoder-Decoder model
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

# Load the saved state dictionary into the model
model.load_state_dict(torch.load(model_path))

#################################################################################################################################
# Part 3: Prediction at the patch level
#################################################################################################################################

# Put the model in evaluation mode
model.eval()

# Adjusting the shapes
normalized_test_input = normalized_test_input.permute(0, 3, 1, 2)
normalized_test_target = normalized_test_target.unsqueeze(1)

# Select a sample from the test set for prediction (takes optical images and releases a canopy height mask)
sample_index = 45  # A sample from the Test set is selected for prediction.
sample_input = normalized_test_input[sample_index]
sample_target = normalized_test_target[sample_index]

# Create a binary mask for valid values (1 for non-NaN, 0 for NaN)
input_mask = ~torch.isnan(sample_input)
target_mask = ~torch.isnan(sample_target)

# Apply the mask to the input and target
sample_input_valid = sample_input * input_mask.unsqueeze(
    0
)  # Apply mask to all channels
sample_target_valid = sample_target * target_mask.unsqueeze(0)

# Reshape the valid samples for prediction
# Reshape the input to (1, channels, height, width)
sample_input_valid = sample_input_valid.to(device, dtype=torch.float32)

# Reshape the target to (1, 1, height, width)
sample_target_valid = sample_target_valid.to(device, dtype=torch.float32)

# Perform prediction
with torch.no_grad():
    predicted_output = model(sample_input_valid)

# Calculate the scaling factor on non-NaN values
scaling_factor = torch.max(
    sample_target_valid[torch.logical_not(torch.isnan(sample_target_valid))]
)

# Rescale the predictions to the original range of target data
predicted_output_rescaled = predicted_output * scaling_factor

#################################################################################################################################
# Part 4: Plotting the predictions at the patch level
#################################################################################################################################

# Convert the tensors to numpy arrays
sample_input_array = sample_input.permute(1, 2, 0).cpu().numpy()
sample_target_array = sample_target.squeeze().cpu().numpy()
predicted_output_array = predicted_output_rescaled.squeeze().cpu().numpy()

# Display the original, target, and predicted images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display the input channels
for i in range(min(sample_input_array.shape[2], 3)):
    axes[i].imshow(sample_input_array[:, :, i], cmap="viridis")
    axes[i].set_title(f"Input Channel {i}")
    axes[i].axis("off")

# Display the target image
axes[1].imshow(sample_target_array, cmap="viridis")
axes[1].set_title("Target")
axes[1].axis("off")

# Display the predicted image
axes[2].imshow(predicted_output_array, cmap="viridis")
axes[2].set_title("Predicted")
axes[2].axis("off")

# Show the plot
plt.show()

# Create a confusion plot with 2D histograms for ground truth and predictions
ground_truth_np = sample_target_valid.cpu().numpy().squeeze()
predictions_np = predicted_output_rescaled.cpu().numpy().squeeze()

# Filter out NaN values
valid_indices = ~np.isnan(ground_truth_np) & ~np.isnan(predictions_np)
filtered_ground_truth = ground_truth_np[valid_indices]
filtered_predictions = predictions_np[valid_indices]

# Create a 2D histogram for the confusion plot
hist2d, xedges, yedges = np.histogram2d(
    filtered_ground_truth, filtered_predictions, bins=50
)

# Plot the confusion plot using Matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(hist2d.T, origin="lower", cmap="viridis")
plt.colorbar(label="Frequency")
plt.xlabel("Ground Truth")
plt.ylabel("Predictions")
plt.title("Confusion Plot")
plt.show()

# Define functions to calculate the metrics
def mse_loss_no_nan(y_true, y_pred):
    valid_mask = ~torch.isnan(y_true)
    mse = torch.mean((y_true[valid_mask] - y_pred[valid_mask]) ** 2)
    return mse

def peak_signal_noise_ratio(y_true, y_pred):
    valid_mask = ~torch.isnan(y_true)
    mse = mse_loss_no_nan(y_true, y_pred)
    psnr = 20 * torch.log10(torch.max(y_true[valid_mask])) - 10 * torch.log10(mse)
    return psnr

def rmse_loss_no_nan(y_true, y_pred):
    valid_mask = ~torch.isnan(y_true)
    mse = mse_loss_no_nan(y_true, y_pred)
    rmse = torch.sqrt(mse)
    return rmse

def mbe_loss_no_nan(y_true, y_pred):
    valid_mask = ~torch.isnan(y_true)
    mbe = torch.mean(y_true[valid_mask] - y_pred[valid_mask])
    return mbe

def bce_loss_no_nan(y_true, y_pred):
    valid_mask = ~torch.isnan(y_true)
    bce = torch.nn.functional.binary_cross_entropy(
        y_pred[valid_mask], y_true[valid_mask]
    )
    return bce


# Calculate and display metrics between target and predicted data
mse_sample = mse_loss_no_nan(sample_target_valid, predicted_output_rescaled)
psnr_sample = peak_signal_noise_ratio(sample_target_valid, predicted_output_rescaled)
rmse_sample = rmse_loss_no_nan(sample_target_valid, predicted_output_rescaled)
mbe_sample = mbe_loss_no_nan(sample_target_valid, predicted_output_rescaled)
bce_sample = bce_loss_no_nan(sample_target_valid, predicted_output_rescaled)

print("Sample Root Mean Squared Error (RMSE):", rmse_sample)
print("Sample Mean Bias Error (MBE):", mbe_sample)
print("Sample Binary Cross-Entropy (BCE):", bce_sample)
print("Sample Mean Squared Error (MSE):", mse_sample)
print("Sample Peak Signal-to-Noise Ratio (PSNR):", psnr_sample)

# Create a list of metric names
metric_names = ["MSE", "RMSE", "MBE"]

# Create a list of the calculated metric values
metric_values = [mse_sample.item(), rmse_sample.item(), mbe_sample.item()]

# Create a bar plot to visualize the metrics
plt.figure(figsize=(10, 6))
plt.bar(metric_names, metric_values, color="blue")
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.title("Metrics Comparison")
plt.show()

# Visualize the difference between target and predicted data
diff = np.abs(sample_target.cpu() - predicted_output_rescaled.cpu().numpy())

# Calculate the range of the absolute difference based on non-NaN values
non_nan_diff = diff[~np.isnan(diff)]
non_nan_diff = np.nan_to_num(non_nan_diff)  # Convert NaN values to zeros

# Calculate the range of the non-NaN absolute difference
diff_range = np.ptp(non_nan_diff)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Select the first (and only) channel from the diff array before visualization
axes[0].imshow(diff[0, 0], cmap="viridis", vmin=0, vmax=diff_range)
axes[0].set_title("Absolute Difference")
axes[0].axis("off")

axes[1].imshow(sample_target[0, :, :].cpu(), cmap="viridis")
axes[1].set_title("Target")
axes[1].axis("off")

axes[2].imshow(predicted_output_rescaled[0, 0, :, :].cpu(), cmap="viridis")
axes[2].set_title("Predicted")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# Creating the cumulative distribution plot
predicted_heights = predicted_output_rescaled.cpu().numpy().flatten()
sorted_heights = np.sort(predicted_heights)
cumulative_distribution = np.arange(1, len(sorted_heights) + 1) / len(sorted_heights)
plt.figure(figsize=(8, 6))
plt.plot(sorted_heights, cumulative_distribution, marker="o", linestyle="-", color="b")
plt.xlabel("Canopy Height Predictions (meters)")
plt.ylabel("Cumulative Distribution")
plt.title("Cumulative Distribution of Canopy Height Predictions")
plt.grid(True)
plt.show()

# Creating the precision-recall curve
prediction_errors = predicted_output_rescaled - sample_target
prediction_errors = (
    prediction_errors.squeeze()
)  # Squeeze the tensor to remove singleton dimensions
prediction_uncertainties = torch.std(
    prediction_errors, dim=(0, 1)
)  # Calculate along dimensions 1 (height) and 2 (width)
precision, recall = [], []
num_total = len(prediction_errors)
for i in range(1, num_total + 1):
    subset_errors = prediction_errors[:i]
    # Remove NaN values from the subset_errors tensor
    subset_errors_non_nan = subset_errors[~torch.isnan(subset_errors)]
    # Calculate the mean squared error for non-NaN values
    mse_non_nan = torch.mean(subset_errors_non_nan**2)
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

#################################################################################################################################
# Part 5: Generate wall-to-wall tree height map
#################################################################################################################################

# -------------------------------------------------------------------------------------------------------------
# Step 1: Reproject to 1m Resolution and Normalize TIFF Files
# -------------------------------------------------------------------------------------------------------------

def calculate_valid_utm_zone(longitude):
    zone = int((longitude + 180) / 6) + 1
    if zone < 1:
        zone = 1
    elif zone > 60:
        zone = 60
    return zone


def generate_data_windows(src, patch_size, overlap):
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
        col_off=0, row_off=0, width=src.width, height=src.height
    )
    offsets = itertools.product(
        range(0, src.width, patch_size - overlap),
        range(0, src.height, patch_size - overlap),
    )
    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=patch_size,
            height=patch_size,
        )
        # Ensure the generated window is within the bounds of the raster
        window = window.intersection(raster_window)

        if window.width > 0 and window.height > 0:
            yield window


def generate_data_windows_from_array(array, patch_size, overlap):
    """
    Generate data windows from a NumPy array with overlapping tiles.

    Parameters:
        array (numpy.ndarray): The source NumPy array.
        patch_size (int): Size of the patches to extract from the array.
        overlap (int): The size of overlap between tiles.

    Yields:
        rasterio.windows.Window: A data window for each tile.
    """
    height, width = array.shape[-2], array.shape[-1]
    raster_window = windows.Window(col_off=0, row_off=0, width=width, height=height)
    offsets = itertools.product(
        range(0, width, patch_size - overlap),
        range(0, height, patch_size - overlap),
    )
    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=patch_size,
            height=patch_size,
        )
        # Ensure the generated window is within the bounds of the array
        window = window.intersection(raster_window)

        if window.width > 0 and window.height > 0:
            yield window


def prediction(input_tiff_path, output_tiff_path, patch_size, overlap):
    """
    Reproject geospatial data from one CRS to another.

    Parameters:
        input_tiff_path (str): Path to the input geospatial TIFF file.
        output_tiff_path (str): Path to save the output reprojected TIFF file.
        patch_size (int): Size of patches used for processing.
        overlap (int): Amount of overlap between patches.
    """
    with rasterio.open(input_tiff_path) as planet_data:
        # Get the affine transformation matrix and dimensions of the input raster
        transform = planet_data.transform
        width = planet_data.width
        height = planet_data.height

        # Define the source CRS (EPSG:3857) and the target CRS (EPSG:4326)
        source_crs = pyproj.CRS("EPSG:3857")
        target_crs = pyproj.CRS("EPSG:4326")

        # Create a transformer to perform the coordinate transformation
        transformer = pyproj.Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )

        # Transform the bounds from EPSG:3857 to EPSG:4326
        left_lon, bottom_lat = transformer.transform(
            planet_data.bounds.left, planet_data.bounds.bottom
        )
        right_lon, top_lat = transformer.transform(
            planet_data.bounds.right, planet_data.bounds.top
        )

        # Calculate the center longitude of the image tile
        center_longitude = (left_lon + right_lon) / 2

        # Calculate the target UTM zone based on the center longitude
        target_utm_zone = calculate_valid_utm_zone(center_longitude)

        # Define the target UTM projection based on the calculated UTM zone
        target_crs = pyproj.CRS(
            f"+proj=utm +zone={target_utm_zone} +datum=WGS84 +units=m +no_defs"
        )

        # Calculate the new affine transformation matrix and dimensions after reprojection
        new_transform, new_width, new_height = calculate_default_transform(
            planet_data.crs,
            target_crs,
            width,
            height,
            *planet_data.bounds,
            resolution=(1, 1),
        )

        # Reproject the planet data using rasterio
        reprojected_data = np.empty(
            (planet_data.count, new_height, new_width), dtype=planet_data.dtypes[0]
        )
        reproject(
            planet_data.read(),
            reprojected_data,
            src_transform=transform,
            src_crs=planet_data.crs,
            dst_transform=new_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )  # Resampling.bilinear

        # Visual verification - Display the reprojected image
        plt.imshow(reprojected_data[0], cmap='viridis')
        plt.title("Reprojected Image")
        plt.show()

        # Define the processed profile
        processed_profile = {
            "driver": "GTiff",
            "width": new_width,
            "height": new_height,
            "count": planet_data.count,
            "dtype": planet_data.dtypes[0],
            "crs": target_crs,
            "transform": new_transform,
        }

        # Now process the normalized TIFF
        for window in generate_data_windows_from_array(
            reprojected_data, patch_size, overlap
        ):
            data = reprojected_data[
                :,
                window.row_off : window.row_off + window.height,
                window.col_off : window.col_off + window.width,
            ]
            data = data / 255
            if data.shape[1] < patch_size or data.shape[2] < patch_size:
                data = np.pad(
                    data,
                    (
                        (0, 0),
                        (0, patch_size - data.shape[1]),
                        (0, patch_size - data.shape[2]),
                    ),
                    mode="reflect",
                )

            # Now perform prediction
            model.to(device)
            model.eval()

            input_tensor = (
                torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                predicted_output = model(input_tensor)

            # Save the predicted output as individual TIFF files
            predicted_window_filename = f"{os.path.splitext(os.path.basename(output_tiff_path))[0]}_predicted_window_{window.row_off}_{window.col_off}.tif"
            predicted_window_path = os.path.join(
                output_folder, predicted_window_filename
            )

            # Reshape the predicted_output tensor to match the window size
            predicted_data = predicted_output[0, 0].cpu().numpy()

            # Use the processed profile instead of src.profile
            with rasterio.open(
                predicted_window_path, "w", **processed_profile, compress="lzw"
            ) as predicted_dst:
                predicted_dst.write(predicted_data, 1)

            print(f"Processed and Predicted on {predicted_window_filename}")

        # Merge the individual TIFF files using rasterio.merge
        predicted_files = glob.glob(os.path.join(output_folder, "*.tif"))
        with rasterio.open(predicted_files[0]) as first_pred:
            profile = first_pred.profile

        # Construct the merged output path using the input TIFF's base name
        merged_output_path = f"{os.path.splitext(os.path.basename(output_tiff_path))[0]}_merged_predicted_output.tif"

        src_files_to_merge = [rasterio.open(fp) for fp in predicted_files]
        mosaic, out_trans = merge(src_files_to_merge)

        profile.update(
            dtype=rasterio.float32,
            count=len(src_files_to_merge),
            compress="lzw",
            nodata=None,
        )

        with rasterio.open(merged_output_path, "w", **profile) as dest:
            dest.write(mosaic)

        print(f"Merged and saved all predictions to {merged_output_path}")

# Paths to input and output folders
input_folder = (
    r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\planet_tiles\Test"
)
output_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet\Predicted_Windows"

# Patch size and overlap
patch_size = 1024
overlap = 64

# Iterate through files in the input folder, reproject, normalize, process windows, and predict
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        prediction(input_path, output_path, patch_size, overlap)

# -------------------------------------------------------------------------------------------------------------
# Step 2: Aggregate Predicted Files and Merge using GDAL
# -------------------------------------------------------------------------------------------------------------

# Paths for predicted TIFFs and aggregated output
predicted_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet\Predicted_Windows"
aggregated_output = "Aggregated_Output_1m.tif"

# List all predicted TIFFs
predicted_files = glob.glob(os.path.join(predicted_folder, "*.tif"))

# Read the first predicted file to get metadata
with rasterio.open(predicted_files[0]) as first_pred:
    profile = first_pred.profile

# Initialize an array for aggregated data
aggregated_data = np.zeros(
    (profile["count"], profile["height"], profile["width"]), dtype=profile["dtype"]
)


def process_pred_file(pred_file):
    with rasterio.open(pred_file) as pred_src:
        return pred_src.read(1)


# Use concurrent.futures for parallel processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_pred_file, predicted_files))

# Aggregate the results
for idx, result in enumerate(results):
    aggregated_data[idx, :, :] = result

# Save the aggregated data using GDAL with LZW compression
driver = gdal.GetDriverByName("GTiff")
aggregated_ds = driver.Create(
    aggregated_output,
    profile["width"],
    profile["height"],
    profile["count"],
    profile["dtype"],
    ["COMPRESS=LZW"],
)
for idx in range(profile["count"]):
    aggregated_ds.GetRasterBand(idx + 1).WriteArray(aggregated_data[idx, :, :])
    aggregated_ds.GetRasterBand(idx + 1).SetNoDataValue(profile["nodata"])
aggregated_ds.SetGeoTransform(profile["transform"])
aggregated_ds.SetProjection(profile["crs"].to_wkt())
aggregated_ds = None  # Close the dataset

# Visual verification - Display the aggregated image
with rasterio.open(aggregated_output) as src:
    plt.imshow(src.read(1), cmap="viridis")
    plt.title("Aggregated Image")
    plt.show()

# -------------------------------------------------------------------------------------------------------------
# Step 3: Convert Resolution to 1-ha grid cell map
# -------------------------------------------------------------------------------------------------------------

# Path for the resampled output TIFF
output_resampled_tiff = "Aggregated_Output_100m.tif"

# Desired grid cell size in square meters (1 hectare)
grid_cell_size = 10000  # 100m x 100m

# Open the aggregated dataset
with rasterio.open(aggregated_output) as src:
    data = src.read()
    transform = src.transform

    # Calculate the number of grid cells in the width and height
    num_cells_width = int(transform.a / grid_cell_size)
    num_cells_height = int(transform.e / grid_cell_size)

    # Calculate the new dimensions and resolution
    new_width = num_cells_width
    new_height = num_cells_height
    new_transform = rasterio.Affine(
        transform.a / num_cells_width,
        transform.b,
        transform.c,
        transform.d,
        transform.e / num_cells_height,
        transform.f,
    )

    # Create a new dataset with the grid cell resolution and LZW compression
    with rasterio.open(
        output_resampled_tiff,
        "w",
        driver="GTiff",
        width=new_width,
        height=new_height,
        count=src.count,
        dtype=data.dtype,
        crs=src.crs,
        transform=new_transform,
        compress="lzw",
    ) as dst:
        for band in range(src.count):
            grid_cell_data = np.zeros((new_height, new_width), dtype=data.dtype)
            rasterio.warp.reproject(
                data[band, :, :],
                grid_cell_data,
                src_transform=transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=rasterio.enums.Resampling.bilinear,
            )
            dst.write(grid_cell_data, band + 1)

# Visual verification - Display the grid cell map
with rasterio.open(output_resampled_tiff) as grid_cell_src:
    plt.imshow(grid_cell_src.read(1), cmap="viridis")
    plt.title("Grid Cell Map (1 Hectare)")
    plt.show()

# -------------------------------------------------------------------------------------------------------------
# Step 4: Studying uncertainty at the pixel level
# -------------------------------------------------------------------------------------------------------------

# List to store prediction arrays
prediction_arrays = []

# Collect predictions (similar to your existing prediction loop)
for filename in os.listdir(output_folder):
    if filename.startswith("predicted_window"):
        prediction_path = os.path.join(output_folder, filename)
        with rasterio.open(prediction_path) as pred_src:
            prediction_arrays.append(pred_src.read(1))

# Calculate pixel-wise variance or standard deviation
pixelwise_variance = np.var(prediction_arrays, axis=0)
pixelwise_stddev = np.std(prediction_arrays, axis=0)

# Visualize uncertainty (using standard deviation)
plt.imshow(pixelwise_stddev, cmap="viridis")
plt.title("Pixel-wise Standard Deviation")
plt.colorbar()
plt.show()

#################################################################################################################################
# Part 5: Generate Visualizations
#################################################################################################################################

# Visualize the generated wall-to-wall tree height map using Datashader
tree_height_map_ds = ds.Canvas(plot_width=tree_height_map.shape[2], plot_height=tree_height_map.shape[3])
agg = tree_height_map_ds.points(tree_height_map[0], agg=ds.by("band"))
agg_shaded = tf.shade(agg, cmap=["blue", "green", "red"], how='linear')
tf.set_background(agg_shaded, "black")
tf.Image(agg_shaded).show()

# Histogram comparing the observed tree heights from the input data (ground truth) with the predicted tree heights from our model
observed_tree_heights = (
    normalized_test_target[:, 0, :, :].cpu().numpy()
)  # Assuming the tree height is in the first channel
predicted_tree_heights = predicted_outputs[:, 0, :, :].cpu().numpy()
plt.hist(
    observed_tree_heights.flatten(), bins=50, alpha=0.5, color="blue", label="Observed"
)
plt.hist(
    predicted_tree_heights.flatten(),
    bins=50,
    alpha=0.5,
    color="orange",
    label="Predicted",
)
plt.xlabel("Tree Height")
plt.ylabel("Frequency")
plt.title("Distribution of Tree Heights: Observed vs. Predicted")
plt.legend()
plt.show()

# Scatter plot with observed tree heights on the x-axis and predicted tree heights on the y-axis.
from scipy.stats import linregress

# Flatten the observed and predicted tree heights arrays
observed_tree_heights_flat = observed_tree_heights.flatten()
predicted_tree_heights_flat = predicted_tree_heights.flatten()

# Find valid indices where both observed and predicted heights are not NaN
valid_indices = np.logical_and(
    ~np.isnan(observed_tree_heights_flat), ~np.isnan(predicted_tree_heights_flat)
)

# Use valid indices to filter the data
observed_tree_heights_valid = observed_tree_heights_flat[valid_indices]
predicted_tree_heights_valid = predicted_tree_heights_flat[valid_indices]

# Calculate R-squared using vectorized operations
observed_mean = np.mean(observed_tree_heights_valid)
predicted_mean = np.mean(predicted_tree_heights_valid)
numerator = np.sum(
    (observed_tree_heights_valid - observed_mean)
    * (predicted_tree_heights_valid - predicted_mean)
)
denominator_observed = np.sqrt(
    np.sum((observed_tree_heights_valid - observed_mean) ** 2)
)
denominator_predicted = np.sqrt(
    np.sum((predicted_tree_heights_valid - predicted_mean) ** 2)
)
r_squared = (numerator / (denominator_observed * denominator_predicted)) ** 2

# Calculate RMSE using vectorized operations
rmse = np.sqrt(
    np.mean((predicted_tree_heights_valid - observed_tree_heights_valid) ** 2)
)

# Create the scatter plot
plt.scatter(
    observed_tree_heights_valid,
    predicted_tree_heights_valid,
    alpha=0.5,
    color="blue",
    label="Data",
)
plt.plot(
    observed_tree_heights_valid,
    observed_tree_heights_valid,
    color="orange",
    label="Regression Line",
)
plt.xlabel("Observed Tree Heights")
plt.ylabel("Predicted Tree Heights")
plt.title(f"Observed vs. Predicted Tree Heights\nR2: {r_squared:.3f}, RMSE: {rmse:.3f}")
plt.legend()
plt.show()

# Scatter plot using hvPlot
data = pd.DataFrame({
    "Observed Tree Heights": observed_tree_heights_flat[valid_indices],
    "Predicted Tree Heights": predicted_tree_heights_flat[valid_indices]
})

# Create a binned scatter plot with a trendline using hvPlot
scatter = data.hvplot.scatter(
    x="Observed Tree Heights",
    y="Predicted Tree Heights",
    alpha=0.5,
    color="blue",
    width=600,
    height=400,
    title=f"Observed vs. Predicted Tree Heights\nR2: {r_squared:.3f}, RMSE: {rmse:.3f}",
)
trendline = hv.Curve([(min(data['Observed Tree Heights']), min(data['Observed Tree Heights'])), 
             (max(data['Observed Tree Heights']), max(data['Observed Tree Heights']))], 
             label="Regression Line", color="orange")

# Combine the scatter plot and trendline
scatter_with_trend = (scatter * trendline).opts(legend_position='top_left')

# Show the plot
scatter_with_trend


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
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x, y, z, cmap="viridis")
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
# Part 6: Generate wall-to-wall variance map
#################################################################################################################################

# Calculate the variance of tree heights across the entire dataset
tree_height_var = np.var(tree_height_map, axis=0)

# Convert the tree height map to a NumPy array with the desired data type
tree_height_var_map_np = tree_height_var.astype(np.float32)

# Save the variance map to a new GeoTIFF file
output_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 80%\Kalimantan_tree_height_variance.tif"
height, width = tree_height_var.shape[1:]
transform = rasterio.transform.from_origin(x_origin, y_origin, resolution, resolution)
crs = rasterio.crs.CRS.from_epsg(4326)

with rasterio.open(
    output_file,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=4,
    dtype=tree_height_var_map_np.dtype,
    crs=crs,
    transform=transform,
) as dst:
    dst.write(tree_height_var_map_np, 1)

# Calculate the value range in meters and print it
value_min = tree_height_var.min()
value_max = tree_height_var.max()
print(f"Value range: {value_min} meters to {value_max} meters")

# Visualisations
# ----------------

# Visualize the generated variance map
plt.imshow(tree_height_var, cmap="coolwarm")
plt.colorbar(label="Tree Height Variance")
plt.title("Tree Height Variance Map")
plt.show()

# NOT WORKING AT THE MOMENT: Visualize the generated variance map using Datashader
variance_map_ds = ds.Canvas(
    plot_width=tree_height_var.shape[1], plot_height=tree_height_var.shape[0]
)
agg = variance_map_ds.tf.shade(tree_height_var, cmap="coolwarm", how="linear")
tf.set_background(agg, "black")
tf.Image(agg).show()

# Histogram comparing the observed tree heights from the input data (ground truth) with the predicted tree heights from our model

observed_tree_heights = normalized_test_target[
    :, 0, :, :
].numpy()  # Assuming the tree height is in the first channel
predicted_tree_heights = predicted_outputs[:, 0, :, :].numpy()
plt.hist(
    observed_tree_heights.flatten(), bins=50, alpha=0.5, color="blue", label="Observed"
)
plt.hist(
    predicted_tree_heights.flatten(),
    bins=50,
    alpha=0.5,
    color="orange",
    label="Predicted",
)
plt.xlabel("Tree Height")
plt.ylabel("Frequency")
plt.title("Distribution of Tree Heights: Observed vs. Predicted")
plt.legend()
plt.show()

# Scatter plot with observed tree heights on the x-axis and predicted tree heights on the y-axis.
from scipy.stats import linregrError

observed_tree_heights_flat = observed_tree_heights.flatten()
predicted_tree_heights_flat = predicted_tree_heights.flatten()
valid_indices = ~np.isnan(observed_tree_heights_flat) & ~np.isnan(
    predicted_tree_heights_flat
)
observed_tree_heights_valid = observed_tree_heights_flat[valid_indices]
predicted_tree_heights_valid = predicted_tree_heights_flat[valid_indices]
r_squared = (
    np.corrcoef(observed_tree_heights_valid, predicted_tree_heights_valid)[0, 1] ** 2
)
rmse = np.sqrt(
    np.mean((predicted_tree_heights_valid - observed_tree_heights_valid) ** 2)
)
plt.scatter(
    observed_tree_heights_valid,
    predicted_tree_heights_valid,
    alpha=0.5,
    color="blue",
    label="Data",
)
plt.plot(
    observed_tree_heights_valid,
    observed_tree_heights_valid,
    color="orange",
    label="Regression Line",
)
plt.xlabel("Observed Tree Heights")
plt.ylabel("Predicted Tree Heights")
plt.title(f"Observed vs. Predicted Tree Heights\nR2: {r_squared:.3f}, RMSE: {rmse:.3f}")
plt.legend()
plt.show()

# Calculate confidence intervals for predicted tree heights
confidence_intervals = (
    1.96 * np.std(predicted_tree_heights, axis=0) / np.sqrt(len(predicted_tree_heights))
)
plt.errorbar(
    observed_tree_heights_flat,
    predicted_tree_heights_flat,
    yerr=confidence_intervals,
    fmt="o",
    markersize=4,
    alpha=0.5,
    color="blue",
    label="Data",
)
plt.plot(
    observed_tree_heights_flat,
    observed_tree_heights_flat,
    color="orange",
    label="Regression Line",
)
plt.xlabel("Observed Tree Heights")
plt.ylabel("Predicted Tree Heights")
plt.title(f"Observed vs. Predicted Tree Heights\nR2: {r_squared:.3f}, RMSE: {rmse:.3f}")
plt.legend()
plt.show()

# Create box plots to visualize the distribution of observed and predicted tree heights
plt.boxplot(
    [observed_tree_heights_flat, predicted_tree_heights_flat],
    labels=["Observed", "Predicted"],
)
plt.ylabel("Tree Heights")
plt.title("Box Plot of Observed and Predicted Tree Heights")
plt.show()

#################################################################################################################################
# Part 6: Compute and save Kalimantan_percent_cover.tif
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
        cell = tree_height_map[row : row + cell_size, col : col + cell_size]
        valid_pixels = cell[mask[row : row + cell_size, col : col + cell_size]]
        percent_cover[row : row + cell_size, col : col + cell_size] = (
            np.count_nonzero(valid_pixels > threshold) / valid_pixels.size * 100
        )

# Save the percent canopy cover map to a new GeoTIFF file (Kalimantan_percent_cover.tif)
output_file = "Kalimantan_percent_cover.tif"
height, width = percent_cover.shape
transform = rasterio.transform.from_origin(
    x_origin, y_origin, cell_size, cell_size
)  # Assuming 1-ha grid with cell_size m resolution
crs = rasterio.crs.CRS.from_epsg(4326)

with rasterio.open(
    output_file,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=percent_cover.dtype,
    crs=crs,
    transform=transform,
) as dst:
    dst.write(percent_cover, 1)

# Visualize the generated percent canopy cover map using a suitable visualization library
plt.imshow(percent_cover, cmap="viridis")
plt.colorbar(label="Percent Canopy Cover (%)")
plt.title("Percent Canopy Cover Map")
plt.show()


#################################################################################################################################
# Part 7: Compute and save Kalimantan_LCA.tif
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
lca_1ha_grid = np.zeros(
    (tree_height_map.shape[0] // cell_size, tree_height_map.shape[1] // cell_size)
)
for row in range(0, tree_height_map.shape[0], cell_size):
    for col in range(0, tree_height_map.shape[1], cell_size):
        cell = lca_clusters[row : row + cell_size, col : col + cell_size]
        lca_1ha_grid[row // cell_size, col // cell_size] = np.sum(cell > 0)

# Compute the percentage of area covered by large trees for each 1-ha cell
lca_percent_cover = (lca_1ha_grid / (cell_size**2)) * 100

# Save the Large Crown Area (LCA) percent cover map to a new GeoTIFF file (Kalimantan_LCA.tif)
output_file_lca = "Kalimantan_LCA.tif"
height, width = lca_percent_cover.shape
transform_lca = rasterio.transform.from_origin(
    x_origin, y_origin, cell_size, cell_size
)  # Assuming 1-ha grid with cell_size m resolution

with rasterio.open(
    output_file_lca,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=lca_percent_cover.dtype,
    crs=crs,
    transform=transform_lca,
) as dst:
    dst.write(lca_percent_cover, 1)

# Visualize the generated Large Crown Area (LCA) percent cover map using a suitable visualization library
plt.imshow(lca_percent_cover, cmap="viridis")
plt.colorbar(label="LCA Percent Cover (%)")
plt.title("Large Crown Area (LCA) Percent Cover Map")
plt.show()


#################################################################################################################################
# Part 8: Compute and save Kalimantan_degradation_index.tif
#################################################################################################################################

# Calculate the Forest Degradation Index (FDI) as FDI = MCH + LCA + PC
# Assuming MCH (mean crown height) is the same as tree_height_map
fdi = tree_height_map + lca_percent_cover + percent_cover

# Save the Forest Degradation Index (FDI) map to a new GeoTIFF file (Kalimantan_degradation_index.tif)
output_file_fdi = "Kalimantan_degradation_index.tif"

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

with rasterio.open(
    output_file_fdi,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=fdi_classes.dtype,
    crs=crs,
    transform=transform,
) as dst:
    dst.write(fdi_classes, 1)

# Visualize the generated Forest Degradation Index (FDI) map using a suitable visualization library
plt.imshow(fdi_classes, cmap="viridis")
plt.colorbar(label="Forest Degradation Index (FDI)")
plt.title("Forest Degradation Index (FDI) Map")
plt.show()

#################################################################################################################################
# Part 9: ABG computation
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
            "c0": -1.803,
            "c1": -0.976,
            "c2": 0.976,
            "c3": 2.673,
            "c4": -0.0299,
        }

    c0 = coefficients["c0"]
    c1 = coefficients["c1"]
    c2 = coefficients["c2"]
    c3 = coefficients["c3"]
    c4 = coefficients["c4"]

    AGB = np.exp(c0 + c1 * E + c2 * np.log(wd) + c3 * np.log(d) + c4 * (np.log(d)) ** 2)
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
    BA_estimated = rho0 * (TCH**rho1) * (1 + rho2 * Cover_resid)
    WD_estimated = rho0 * (TCH**rho1)
    ACD = BA_estimated**rho2 * WD_estimated**rho3
    return ACD


# Implementation of Monte Carlo simulations to estimate uncertainty:

# Define uncertain parameters and their distributions
num_simulations = 1000

# Generate random samples for the coefficients of AGB and ACD models
uncertain_agb_coeff_samples = np.random.normal(
    mean_agb_coeffs, std_agb_coeffs, (num_simulations, 5)
)
uncertain_acd_rho_samples = np.random.normal(
    mean_acd_rho, std_acd_rho, (num_simulations, 4)
)

# Placeholder for storing predicted values from simulations
predicted_agb_values = []
predicted_acd_values = []

# Run simulations for AGB
for agb_coeffs in uncertain_agb_coeff_samples:
    AGB_prediction = calculate_aboveground_biomass(d, wd, E, agb_coeffs)
    predicted_agb_values.append(AGB_prediction)

# Run simulations for ACD
for acd_rho_values in uncertain_acd_rho_samples:
    ACD_prediction = calculate_aboveground_carbon_density(
        TCH, Cover_resid, BA, WD, acd_rho_values
    )
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
diameters = [
    15,
    20,
    25,
]  # Diameter of boles measured at diameter at breast height or above the buttress (cm)
wood_densities = [0.6, 0.55, 0.65]  # Wood densities (g cm^-3) for each tree species
plot_area = 1  # Area of the plot in hectares (ha)

# Calculate plot-level aboveground biomass (AGBp) using the sub-sampling strategy
AGB_plot_level_sub_sampling = calculate_agb_plot_level(
    diameters, wood_densities, plot_area
)

# Calculate plot-level aboveground biomass (AGBp) using the tree-level strategy
AGB_tree_level = [
    calculate_agb_tree_level(d, wd) for d, wd in zip(diameters, wood_densities)
]
AGB_plot_level_tree_level = np.sum(AGB_tree_level) / (
    plot_area * 0.0001
)  # Convert area to m²

# Save the plot-level AGB values in a GeoTIFF file (Kalimantan_aboveground_biomass.tif)
output_file_agb = "Kalimantan_aboveground_biomass.tif"
height, width = 1, 1
transform_agb = rasterio.transform.from_origin(
    0, 0, 1, 1
)  # Assuming 1-ha grid with 1 m resolution
crs_agb = rasterio.crs.CRS.from_epsg(4326)  # Assuming WGS84 coordinate reference system

with rasterio.open(
    output_file_agb,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="float32",
    crs=crs_agb,
    transform=transform_agb,
) as dst:
    dst.write(AGB_plot_level_tree_level, 1)

# Print the range of values measured in Mg biomass/ha
print(
    f"Plot-level AGB using sub-sampling strategy: {AGB_plot_level_sub_sampling} Mgha^-1"
)
print(f"Plot-level AGB using tree-level strategy: {AGB_plot_level_tree_level} Mgha^-1")
