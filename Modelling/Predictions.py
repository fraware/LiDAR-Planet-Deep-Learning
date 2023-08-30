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
input_folder = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\planet_tiles\Processed Planet"  # Optical
target_folder = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\LiDAR\Processed LiDAR"  # LiDAR

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
# Part 3: Random Multiple Prediction at the patch level
#################################################################################################################################

# Put the model in evaluation mode
model.eval()

# Adjusting the shapes
normalized_test_input = normalized_test_input.permute(0, 3, 1, 2)
normalized_test_target = normalized_test_target.unsqueeze(1)

# Number of random patches to predict
num_patches = 5

for _ in range(num_patches):
    # Select a random sample from the test set for prediction (takes optical images and releases a canopy height mask)
    sample_index = np.random.randint(len(normalized_test_input))
    sample_input = normalized_test_input[sample_index]
    sample_target = normalized_test_target[sample_index]

    # Create a binary mask for valid values (1 for non-NaN, 0 for NaN)
    input_mask = ~torch.isnan(sample_input)
    target_mask = ~torch.isnan(sample_target)

    # Apply the mask to the input and target
    sample_input_valid = sample_input * input_mask.unsqueeze(0)  # Apply mask to all channels
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

    # -------------------------------------------------------------------------------------------------------------
    # Plotting the predictions for each patch
    # -------------------------------------------------------------------------------------------------------------

    # Convert the tensors to numpy arrays
    sample_input_array = sample_input.permute(1, 2, 0).cpu().numpy()
    sample_target_array = sample_target.squeeze().cpu().numpy()
    predicted_output_array = predicted_output_rescaled.squeeze().cpu().numpy()

    # Visualize the difference between target and predicted data
    diff = np.abs(sample_target.cpu() - predicted_output_rescaled.cpu().numpy())

    # Calculate the range of the absolute difference based on non-NaN values
    non_nan_diff = diff[~np.isnan(diff)]
    non_nan_diff = np.nan_to_num(non_nan_diff)  # Convert NaN values to zeros

    # Calculate the range of the non-NaN absolute difference
    diff_range = np.ptp(non_nan_diff)

    # Display the original, target, and predicted images along with the difference range for each patch
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # Display the input channels
    for i in range(min(sample_input_array.shape[2], 3)):
        axes[0, i].imshow(sample_input_array[:, :, i], cmap="terrain")
        axes[0, i].set_title(f"Input Channel {i+1}")
        axes[0, i].axis("off")

    # Display the target image
    axes[0, 3].imshow(sample_target_array, cmap="viridis")
    axes[0, 3].set_title("Target")
    axes[0, 3].axis("off")

    # Display the predicted image
    axes[0, 4].imshow(predicted_output_array, cmap="terrain")
    axes[0, 4].set_title("Predicted")
    axes[0, 4].axis("off")

    # Show the difference range
    axes[0, 5].imshow(diff, cmap="coolwarm")
    axes[0, 5].set_title("Absolute Difference")
    axes[0, 5].axis("off")

    # Histogram of tree height values for each sample
    axes[1, 0].hist(sample_target_array[~np.isnan(sample_target_array)], bins=20, color='blue', alpha=0.5, label='Target')
    axes[1, 0].hist(predicted_output_array[~np.isnan(predicted_output_array)], bins=20, color='red', alpha=0.5, label='Predicted')
    axes[1, 0].set_title("Tree Height Histogram")
    axes[1, 0].legend()

    # Show the plot for each patch
    plt.show()

    # Print the calculated difference range for each patch
    print(f"Range of Absolute Difference (Patch {sample_index}): {diff_range}")


#################################################################################################################################
# Part 4: Prediction at the patch level
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


# Histogram comparing the observed tree heights from the input data (ground truth) with the predicted tree heights from our model
observed_tree_heights = sample_target_valid.cpu().numpy()
predicted_tree_heights = predicted_output_rescaled.cpu().numpy()
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

# Alternative scatter plot using hvPlot
data = pd.DataFrame(
    {
        "Observed Tree Heights": observed_tree_heights_flat[valid_indices],
        "Predicted Tree Heights": predicted_tree_heights_flat[valid_indices],
    }
)

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
trendline = hv.Curve(
    [
        (min(data["Observed Tree Heights"]), min(data["Observed Tree Heights"])),
        (max(data["Observed Tree Heights"]), max(data["Observed Tree Heights"])),
    ],
    label="Regression Line",
    color="orange",
)

# Combine the scatter plot and trendline
scatter_with_trend = (scatter * trendline).opts(legend_position="top_left")

# Show the plot
scatter_with_trend


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
            resampling=Resampling.bilinear,
        )  # Resampling.nearest

        # # Visual verification - Display the reprojected image
        # plt.imshow(reprojected_data[0], cmap='viridis')
        # plt.title("Reprojected Image")
        # plt.show()

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
            # model = unet_model(input_shape=(4, patch_size, patch_size), n_classes=1)
            model.to(device)
            model.eval()

            input_tensor = (
                torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                predicted_output = model(input_tensor)

            # Create a unique processed_profile for each predicted window
            processed_profile = {
                "driver": "GTiff",
                "width": window.width,
                "height": window.height,
                "count": predicted_output.shape[1],
                #"dtype": planet_data.dtypes[0],
                "crs": target_crs,
                "transform": new_transform,
                # "transform": from_origin(
                #     new_transform.c, new_transform.f, new_transform.a, new_transform.e
                # ),
            }

            # Save the predicted output as individual TIFF files
            predicted_window_filename = f"{os.path.splitext(os.path.basename(output_tiff_path))[0]}_predicted_window_{window.row_off}_{window.col_off}.tif"
            predicted_window_path = os.path.join(
                output_folder, predicted_window_filename
            )

            # Reshape the predicted_output tensor to match the window size
            predicted_data = predicted_output[0, 0].cpu().numpy()

            # print("Predicted Data Summary:")
            # print(f"Min Value: {predicted_data.min()}")
            # print(f"Max Value: {predicted_data.max()}")

            # Use the processed profile instead of src.profile
            with rasterio.open(
                predicted_window_path, "w", **processed_profile, compress="lzw"
            ) as predicted_dst:
                predicted_dst.write(predicted_data, 1)

            print(f"Processed and Predicted on {predicted_window_filename}")


def merge_predicted_tiff_files_vrt(
    output_folder, output_tiff_path, output_folder_tiles
):
    # Get a list of all predicted TIFF files in the output folder
    predicted_files = glob.glob(os.path.join(output_folder, "*.tif"))

    # Create a VRT file that includes all the predicted TIFF files
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    base_filename = os.path.splitext(os.path.basename(output_tiff_path))[0]
    vrt_file = os.path.join(output_folder, f"{base_filename}_{timestamp}_predicted_files.vrt")

    # Use gdalbuildvrt to create the VRT file
    gdalbuildvrt_command = ["gdalbuildvrt", "-input_file_list", vrt_file]
    gdalbuildvrt_command.extend(predicted_files)
    subprocess.run(gdalbuildvrt_command)

    # Use gdal_translate to merge the VRT file into a single output TIFF
    merged_output_file = f"{base_filename}_{timestamp}_merged_predicted_output.tif"
    merged_output_path = os.path.join(output_folder_tiles, merged_output_file)

    # Run gdal_translate command to create the merged TIFF
    gdal_translate_command = [
        "gdal_translate",
        "-of",
        "GTiff",
        vrt_file,
        merged_output_path,
    ]
    subprocess.run(gdal_translate_command)

    print(f"Merged and saved predictions to {merged_output_path}")

    # Visual verification of the merged output
    with rasterio.open(merged_output_path) as merged_pred:
        plt.figure(figsize=(10, 8))
        plt.imshow(merged_pred.read(1), cmap="viridis")
        plt.title("Merged Predicted Output")
        plt.colorbar()
        plt.show()


# Paths to input and output folders
input_folder = (
    r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\planet_tiles\Test"
)
output_folder = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\planet_tiles\Processed Planet\Predicted_Windows"

# Patch size and overlap
patch_size = 1024
overlap = 64

# Iterate through files in the input folder, reproject, normalize, process windows, and predict
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        prediction(input_path, output_path, patch_size, overlap)
        merge_predicted_tiff_files_vrt(output_folder, output_path, output_folder_tiles)

# -------------------------------------------------------------------------------------------------------------
# Step 2: Aggregate Predicted Files and Merge using GDAL
# -------------------------------------------------------------------------------------------------------------

# Paths for predicted TIFFs and aggregated output
predicted_folder = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\planet_tiles\Processed Planet\Predicted_Windows"
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


# -------------------------------------------------------------------------------------------------------------
#  Step 4: Wall-to-wall Map Visualisations
# -------------------------------------------------------------------------------------------------------------

# Define the paths to the data folders
wall_to_wall_map_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Output\wall_to_wall_map_kalimantan.tif"

# Open the wall-to-wall map file with rasterio
with rasterio.open(wall_to_wall_map_file) as src:
    # Read the wall-to-wall map data as a numpy array
    tree_height_map = src.read(1)

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
    predicted_tree_heights = predicted_output[:, 0, :, :].cpu().numpy()
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

    # Alternative scatter plot using hvPlot
    data = pd.DataFrame(
        {
            "Observed Tree Heights": observed_tree_heights_flat[valid_indices],
            "Predicted Tree Heights": predicted_tree_heights_flat[valid_indices],
        }
    )

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
    trendline = hv.Curve(
        [
            (min(data["Observed Tree Heights"]), min(data["Observed Tree Heights"])),
            (max(data["Observed Tree Heights"]), max(data["Observed Tree Heights"])),
        ],
        label="Regression Line",
        color="orange",
    )

    # Combine the scatter plot and trendline
    scatter_with_trend = (scatter * trendline).opts(legend_position="top_left")

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

    # Kernel Density Estimation (KDE) Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(tree_height_map.flatten(), shade=True)
    plt.xlabel("Tree Height (m)")
    plt.ylabel("Density")
    plt.title("Kernel Density Estimation (KDE) Plot of Tree Heights")
    plt.show()

# -------------------------------------------------------------------------------------------------------------
#  Step 5: Studying uncertainty at the pixel level
# -------------------------------------------------------------------------------------------------------------

# List to store prediction arrays
prediction_arrays = []

# Collect predictions (similar to your existing prediction loop)
output_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output"

for filename in os.listdir(output_folder):
    if filename.startswith("predicted_window"):
        prediction_path = os.path.join(output_folder, filename)
        with rasterio.open(prediction_path) as pred_src:
            prediction_arrays.append(pred_src.read(1))

# Number of Monte Carlo simulations
num_simulations = 1000

# List to store pixel-wise standard deviations from each simulation
pixelwise_stddev_simulations = []

# Perform Monte Carlo simulations
for _ in range(num_simulations):
    # Resample predictions with replacement
    resampled_predictions = [np.random.choice(pred_array.flatten(), size=pred_array.size, replace=True)
                             for pred_array in prediction_arrays]

    # Calculate pixel-wise standard deviation for the resampled predictions
    resampled_stddev = np.std(resampled_predictions, axis=0)

    # Append the resampled standard deviation to the list
    pixelwise_stddev_simulations.append(resampled_stddev)

# Calculate the mean and confidence intervals (e.g., 95% CI) from the simulations
mean_stddev = np.mean(pixelwise_stddev_simulations, axis=0)
lower_percentile = np.percentile(pixelwise_stddev_simulations, 2.5, axis=0)
upper_percentile = np.percentile(pixelwise_stddev_simulations, 97.5, axis=0)

# Visualize uncertainty (using the mean and confidence intervals)
plt.figure(figsize=(10, 6))
plt.imshow(mean_stddev, cmap="viridis")
plt.title("Monte Carlo Simulations of Pixel-wise Standard Deviation")
plt.colorbar()
plt.show()

# Visualize the 95% confidence intervals
plt.figure(figsize=(10, 6))
plt.imshow(mean_stddev, cmap="viridis")
plt.fill_between(range(mean_stddev.shape[1]), lower_percentile, upper_percentile, color='red', alpha=0.5)
plt.title("95% Confidence Intervals of Pixel-wise Standard Deviation")
plt.colorbar()
plt.show()

# -------------------------------------------------------------------------------------------------------------
#  Step 6: Generate wall-to-wall variance map
# -------------------------------------------------------------------------------------------------------------

# Uncertainty at the pixel level

# Define the paths to the data folders
wall_to_wall_map_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Output\wall_to_wall_map_kalimantan.tif"

# Open the wall-to-wall map file with rasterio
with rasterio.open(wall_to_wall_map_file) as src:
    # Read the wall-to-wall map data as a numpy array
    wall_to_wall_map = src.read(1)

    # Calculate the variance of tree heights
    tree_height_var = np.var(wall_to_wall_map, axis=0)

    # Convert the tree height map to a NumPy array with the desired data type
    tree_height_var_map_np = tree_height_var.astype(np.float32)

# Save the variance map to a new GeoTIFF file
output_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\First Model - July 23 - U-Net - 5000 epochs\Within 80%\Kalimantan_tree_height_variance.tif"
height, width = tree_height_var.shape[1:]
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
) as dst:
    dst.write(tree_height_var_map_np, 1)

# Calculate the value range in meters and print it
value_min = tree_height_var.min()
value_max = tree_height_var.max()
print(f"Value range: {value_min} meters to {value_max} meters")

# -------------------------------------------------------------------------------------------------------------
#  Step 7: Wall-to-wall Variance Map Visualisations
# -------------------------------------------------------------------------------------------------------------

# Visualize the generated variance map
plt.imshow(tree_height_var, cmap="coolwarm")
plt.colorbar(label="Tree Height Variance")
plt.title("Tree Height Variance Map")
plt.show()

# Visualize the generated variance map using Datashader
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
predicted_tree_heights = predicted_output[:, 0, :, :].numpy()
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

