# The purpose of this file is to train and validate our models. 

#################################################################################################################################
# Part 1: Importing the necessary libraries
#################################################################################################################################

import os
import numpy as np
import rasterio
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
from torch.utils.data import Dataset, DataLoader
import skimage.util as sk_util
import plotly.express as px
from skimage.transform import resize
import tifffile as tiff
import imgaug.augmenters as iaa
import glob
import torch
import hvplot.pandas
from bokeh.sampledata.penguins import data as df
torch.cuda.empty_cache()
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_tensor
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
print("PyTorch version:", torch.__version__)  # PyTorch version: 2.0.1+cu118
import fastai.vision.all
from torchsummary import summary
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
import xarray as xr
import rioxarray as rxr
import rasterio as rio
from rasterio.windows import Window
from torch.utils.tensorboard import SummaryWriter
from skimage.util import view_as_windows
import tifffile as tiff
import concurrent.futures
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import rasterio.crs as rcrs
from rasterio.crs import CRS
import xarray as xr

#################################################################################################################################
# Part 2: Defining the necessary elements
#################################################################################################################################

# Define the paths to the input and target data folders
input_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Processed Planet"  # Optical
target_folder = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR"  # LiDAR

# Load the preprocessed data
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

#################################################################################################################################
# Part 3: Model Definition
#################################################################################################################################

def unet_model(n_channels, n_classes):
    """
    Create a U-Net model for semantic segmentation.

    Args:
        n_channels (int): Number of input channels (e.g., 3 for RGB images).
        n_classes (int): Number of output classes (e.g., number of segmentation classes).

    Returns:
        nn.Module: U-Net model with specified number of input and output channels.

    """
    class DoubleConv(nn.Module):
        """(convolution => [BN] => ReLU) * 2"""

        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)


    class Down(nn.Module):
        """Downscaling with maxpool then double conv"""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

        def forward(self, x):
            return self.maxpool_conv(x)


    class Up(nn.Module):
        """Upscaling then double conv"""

        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()

            # if bilinear, use the normal convolutions to reduce the number of channels
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)


    class OutConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(OutConv, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, n_channels, n_classes, bilinear=False):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear

            self.inc = (DoubleConv(n_channels, 64))
            self.down1 = (Down(64, 128))
            self.down2 = (Down(128, 256))
            self.down3 = (Down(256, 512))
            factor = 2 if bilinear else 1
            self.down4 = (Down(512, 1024 // factor))
            self.up1 = (Up(1024, 512 // factor, bilinear))
            self.up2 = (Up(512, 256 // factor, bilinear))
            self.up3 = (Up(256, 128 // factor, bilinear))
            self.up4 = (Up(128, 64, bilinear))
            self.outc = (OutConv(64, n_classes))

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits

        def use_checkpointing(self):
            self.inc = torch.utils.checkpoint(self.inc)
            self.down1 = torch.utils.checkpoint(self.down1)
            self.down2 = torch.utils.checkpoint(self.down2)
            self.down3 = torch.utils.checkpoint(self.down3)
            self.down4 = torch.utils.checkpoint(self.down4)
            self.up1 = torch.utils.checkpoint(self.up1)
            self.up2 = torch.utils.checkpoint(self.up2)
            self.up3 = torch.utils.checkpoint(self.up3)
            self.up4 = torch.utils.checkpoint(self.up4)
            self.outc = torch.utils.checkpoint(self.outc)

    model = UNet(n_channels, n_classes).to(device)
    return model


def CNN(input_shape, device):
    """
    Creates a Convolutional Neural Network (CNN) model for a regression task.

    Args:
        input_shape (tuple): The shape of the input data (channels, height, width).
        device: The device on which the model will be placed (e.g., 'cuda' or 'cpu').

    Returns:
        torch.nn.Module: A CNN model for regression.
    """

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
                test_input = torch.zeros(
                    1, *input_shape
                )  # Create a dummy tensor with batch size 1
                enc_features = self.encoder(test_input)
                self.encoder_output_size = enc_features.view(
                    enc_features.size(0), -1
                ).shape[1]

            # Classifier with fully connected layers
            self.classifier = nn.Sequential(
                nn.Linear(self.encoder_output_size, 256),
                nn.ReLU(inplace=True),
                nn.Linear(
                    256, 1
                ),  # Output only 1 value for regression task (height prediction)
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

            # Custom encoder with input_shape[0] input channels
            self.encoder = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1).to(
                    device
                ),
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

            # Replace the fc layer with a linear layer for regression
            self.fc = nn.Linear(
                512, 1
            )  # Output only 1 value for regression (height prediction)
            # No activation function (no nn.Sigmoid()) for regression

        def forward(self, x):
            enc_features = self.encoder(x)
            output = self.fc(
                enc_features.view(enc_features.size(0), -1)
            )  # Flatten the output for regression
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

            # Custom encoder with input_shape[0] input channels
            self.encoder = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1).to(
                    device
                ),
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

            # Decoder with upsampling layers
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.Conv2d(
                    64, 1, 1
                ),  # Output 1 channel for regression (predicted height)
                # No activation function (no nn.Sigmoid()) for regression
            )

        def forward(self, x):
            enc_features = self.encoder(x)
            dec_features = self.decoder(enc_features)
            return dec_features

    model = EncoderDecoder().to(device)
    return model


# Define the input shape of the U-Net model
input_shape = smoothed_train_input.shape[1:] #4
n_classes=1

# Create the U-Net model
model = unet_model(input_shape, n_classes)

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

# Visualize the execution graph using Torchviz
y = model(smoothed_train_input)
make_dot(y.mean(), params=dict(model.named_parameters()))

# Save the model
torch.save(model.state_dict(), model_path)
print("Model saved successfully.")

#################################################################################################################################
# Part 4: Loss Functions Definition
#################################################################################################################################

mse_loss = nn.MSELoss(reduction="mean")
bce_loss = nn.BCELoss(reduction="mean")

def mse_loss_no_nan(output, target):
    """
    Custom loss function for Mean Squared Error (MSE) loss, excluding NaN values.

    :param output: torch tensor representing the model's output
    :param target: torch tensor representing the target values
    :return: loss value if shapes match, otherwise a placeholder tensor with value 0
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


def mbe_loss_no_nan(output, target):  # to check ! I still have NaN.
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


def bce_loss_no_nan(output, target):  # to check ! I still have NaN.
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
        return torch.tensor(
            0.0, device=target.device, dtype=target.dtype
        )  # Return 0 loss if all loss values are NaN

    return valid_loss_values


def peak_signal_noise_ratio(y_true, y_pred):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the true and predicted values.

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
    if torch.any(torch.isnan(y_true_filtered)) or torch.any(
        torch.isnan(y_pred_filtered)
    ):
        raise ValueError("Input tensors contain NaN or Inf values.")

    # Calculate the mean squared error (MSE) between the filtered y_true and y_pred
    mse = torch.mean((y_true_filtered - y_pred_filtered) ** 2)

    # Calculate the peak signal-to-noise ratio (PSNR)
    psnr = -10 * torch.log10(mse)

    return psnr.item()

#################################################################################################################################
# Part 5: Model Training
#################################################################################################################################

# Define the batch size and number of epochs
batch_size = 16
epochs = 50  

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the current time
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/gradient_tape/" + current_time + "/train"
test_log_dir = "logs/gradient_tape/" + current_time + "/test"

# Start time measurement for training
start_time = time.time()

history = {"loss": [], "val_loss": []}

# Ensure train_target_patches has the correct shape (4256, 1, 128, 128)
train_target_patches = train_target_patches.unsqueeze(1).to(device)  # Not necessary

# Split the input and target data into batches while preserving order
num_elements = len(train_input_patches)  # Total number of elements in the tensors

input_batches = []
target_batches = []

for start_idx in range(0, num_elements, batch_size):
    end_idx = min(start_idx + batch_size, num_elements)
    input_batch = train_input_patches[start_idx:end_idx]
    target_batch = train_target_patches[start_idx:end_idx]

    input_batches.append(input_batch)
    target_batches.append(target_batch)

# Convert the batched lists to tensors
input_batches = [batch.to(device) for batch in input_batches]
target_batches = [batch.to(device) for batch in target_batches]

# Define the shape of normalized_val_input
normalized_val_input = normalized_val_input.permute(0, 3, 1, 2)

# Define the saving interval
save_interval = 5

# Define a learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

for epoch in range(epochs):
    model.train()

    all_indices = range(len(input_batches))

    total_loss = 0.0

    for batch_idx in tqdm(all_indices, desc=f"Epoch {epoch + 1}/{epochs}", ncols=80):
        optimizer.zero_grad()

        # Process the batch_input and batch_target tensors per batch
        batch_input = (
            input_batches[batch_idx].clone().detach().to(device, dtype=torch.float32)
        )
        batch_target = (
            target_batches[batch_idx].clone().detach().to(device, dtype=torch.float32)
        )

        # Check if the batch size matches the desired batch size (16).
        if batch_input.shape[0] != batch_size:
            continue  # Skip this batch and move to the next one

        # Check if the batch size matches the desired batch size (16).
        if batch_target.shape[0] != batch_size:
            continue  # Skip this batch and move to the next one

        # Transpose the input tensors to (batch_size, channels, height, width)
        batch_input = batch_input.permute(0, 3, 1, 2)
        #batch_input = batch_input.permute(0, 3, 1, 2)

        # Forward pass
        output = model(batch_input)

        # Calculate the loss for this batch
        loss = mse_loss_no_nan(output, batch_target)

        # Accumulate the batch loss to the total loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    # Calculate the average loss for the epoch
    avg_epoch_loss = total_loss / len(input_batches)

    # Print the average loss for the epoch
    print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}")

    # Step the learning rate scheduler based on the validation loss
    scheduler.step(avg_epoch_loss)

    # Save the model weights every save_interval epochs
    if (epoch + 1) % save_interval == 0:
        torch.save(model.state_dict(), f'u-net_model_weights_epoch{epoch + 1}.pt')

    # Display predictions on one random image from the batch
    model.eval()
    with torch.no_grad():
        # Select one random image from the batch
        random_image_idx = random.choice(random_indices)
        random_input = input_batches[random_image_idx].to(device, dtype=torch.float32)
        random_target = target_batches[random_image_idx].to(device, dtype=torch.float32)

        predicted_output = model(random_input.permute(0,3,1,2))

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(random_input[1, :, :, :3].cpu().numpy(), vmax=0.4)
        plt.title("Input Image")

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_output[1, :, :, :].permute(1,2,0).cpu().numpy(), vmax=0.3)
        plt.title("Predicted Output")

        plt.subplot(1, 3, 2)
        plt.imshow(random_target[1, :, :, :].permute(1,2,0).cpu().numpy(), vmax=0.65)
        plt.title("Target Image")

        plt.show()

# Extract the heights of trees from random_target and predicted_output
random_heights = random_target[1, :, :, 1].cpu().numpy()  # Green Band
predicted_heights = predicted_output[1, :, :, 1].cpu().numpy()  # Green Band

# Plot histograms of tree heights
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(random_heights.flatten(), bins=20, range=(0, 20), color='blue', alpha=0.5, label='Random Target')
plt.xlabel("Height (meters)")
plt.ylabel("Frequency")
plt.title("Height Distribution in Random Target")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(predicted_heights.flatten(), bins=20, range=(0, 20), color='green', alpha=0.5, label='Predicted Output')
plt.xlabel("Height (meters)")
plt.ylabel("Frequency")
plt.title("Height Distribution in Predicted Output")
plt.legend()

plt.show()

#################################################################################################################################
# Part 6: Model Traning & Validation
#################################################################################################################################

# Define gradient accumulation steps
gradient_accumulation_steps = 4

# Define a batch size for validation
val_batch_size = 8

# Lists to store evaluation metrics
rmse_values = []
psnr_values = []
mbe_values = []
bce_values = []

for epoch in range(epochs):
  
    model.train()
  
    for batch_idx in range(len(input_batches)):
        total_loss = 0.0
        # Data preprocessing and preparation
        optimizer.zero_grad()

        # Move the data to the GPU
        batch_input = (
            input_batches[batch_idx].clone().detach().to(device, dtype=torch.float32)
        )
        batch_target = (
            target_batches[batch_idx].clone().detach().to(device, dtype=torch.float32)
        )

        # Transpose the input tensors to (batch_size, channels, height, width)
        batch_input = batch_input.permute(0, 3, 1, 2)

        # Forward pass
        output = model(batch_input)
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
            print(
                f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx + 1}/{len(input_batches)}] Loss: {total_loss:.4f}"
            )
            total_loss = 0.0

    # Validation at the end of the epoch
    model.eval()
    val_loss_sum = 0.0

    with torch.no_grad():
        for val_batch_idx in range(0, len(normalized_val_input), val_batch_size):
            val_batch_input = normalized_val_input[
                val_batch_idx : val_batch_idx + val_batch_size
            ].to(device)
            val_batch_target = (
                normalized_val_target[val_batch_idx : val_batch_idx + val_batch_size]
                .unsqueeze(1)
                .to(device)
            )

            val_output = model(val_batch_input)
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
            bce = bce_loss_no_nan(val_output, val_batch_target)

            # Append metrics to lists
            rmse_values.append(rmse.item())
            psnr_values.append(psnr)
            mbe_values.append(mbe.item())
            bce_values.append(bce.item())

        # Average validation loss over batches
        avg_val_loss = val_loss_sum / (len(normalized_val_input) // val_batch_size)

        print(f"Epoch [{epoch + 1}/{epochs}] Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation RMSE: {np.mean(rmse_values):.4f}")
        print(f"Validation PSNR: {np.mean(psnr_values):.4f}")
        print(f"Validation MBE: {np.mean(mbe_values):.4f}")
        print(f"Validation BCE: {np.mean(bce_values):.4f}")

        # Clear the lists for the next epoch
        rmse_values.clear()
        psnr_values.clear()
        mbe_values.clear()
        bce_values.clear()

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
torch.save(model.state_dict(), model_path)
print("Model saved successfully.")

#################################################################################################################################
# Part 7: Model Evaluation Visualization
#################################################################################################################################

# -------------------------------------------------------------------------------------------------------------
# Visualization Functions
# -------------------------------------------------------------------------------------------------------------

def visualize_scalars(scalar_values, scalar_names, epochs=epochs):
    """
    Visualize scalar values over epochs.

    :param scalar_values: list of scalar values over time
    :param scalar_names: list of names corresponding to the scalar values
    :param epochs: number of epochs
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

# Calculate metrics only on non-NaN values
valid_rmse_values = [value for value in rmse_values if not np.isnan(value)]
valid_psnr_values = [value for value in psnr_values if not np.isnan(value)]
valid_mbe_values = [value for value in mbe_values if not np.isnan(value)]
valid_bce_values = [value for value in bce_values if not np.isnan(value)]

print(f"Root Mean Squared Error (RMSE): {np.nanmean(valid_rmse_values):.4f}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {np.nanmean(valid_psnr_values):.4f}")
print(f"Mean Bias Error (MBE): {np.nanmean(valid_mbe_values):.4f}")
print(f"Binary Cross-Entropy (BCE): {np.nanmean(valid_bce_values):.4f}")

# Visualize evaluation metrics after the loop
scalar_values = [valid_rmse_values, valid_psnr_values, valid_mbe_values]
scalar_names = ["RMSE", "PSNR", "MBE"]
visualize_scalars(scalar_values, scalar_names)

# Visualize train_loss values over time
train_loss = [history["loss"]]
scalar_values = [train_loss]
scalar_names = ["Training Loss"]
visualize_scalars(
    scalar_values, scalar_names
)

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
