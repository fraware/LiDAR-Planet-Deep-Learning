# The purpose of this file is to create and apply cloud masks to Planet images. 

#####################################################################################################################
# Part 1: Importing necessary libraries
#####################################################################################################################

import os
import glob
import numpy as np
import tensorflow as tf
import rasterio
import random
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.features import geometry_mask
from osgeo import gdal
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    Dropout,
    UpSampling2D,
    Concatenate,
    Conv2DTranspose,
)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

#####################################################################################################################
# Part 2: Defining the functions
#####################################################################################################################

def EncoderMiniBlock(
    inputs, n_filters=32, dropout_prob=0.25, max_pooling=True, kernel_size=(3, 3)
):
    """
    Build a mini-block for the encoder part of the U-Net.

    Parameters:
        inputs (Tensor): Input tensor to the mini-block.
        n_filters (int): Number of filters in the convolutional layers.
        dropout_prob (float): Dropout probability for the mini-block.
        max_pooling (bool): Whether to apply max pooling.
        kernel_size (tuple): Size of the convolutional kernel.

    Returns:
        tuple or Tensor: Output tensor(s) of the mini-block.
    """
    # Convolutional layer
    conv = Conv2D(n_filters, kernel_size=kernel_size, padding="same")(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)

    # Second convolutional layer
    conv = Conv2D(n_filters, kernel_size=kernel_size, padding="same")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)

    # Apply max pooling and dropout (if specified)
    if max_pooling:
        conv_pool = MaxPool2D(pool_size=(2, 2))(conv)
        if dropout_prob > 0:
            conv_pool = Dropout(dropout_prob)(conv_pool)
        return conv_pool, conv
    else:
        return conv


def DecoderMiniBlock_modif(
    prev_layer_input, prev_layer_nonpool_input, n_filters=32, kernel_size=(3, 3)
):
    """
    Build a mini-block for the decoder part of the U-Net.

    Parameters:
        prev_layer_input (Tensor): Input tensor to the previous layer.
        prev_layer_nonpool_input (Tensor): Input tensor to the previous layer without pooling.
        n_filters (int): Number of filters in the convolutional layers.
        kernel_size (tuple): Size of the convolutional kernel.

    Returns:
        Tensor: Output tensor of the mini-block.
    """
    # Convolutional layers
    conv_transpose = Conv2DTranspose(
        n_filters, kernel_size=kernel_size, strides=(2, 2), padding="same"
    )(prev_layer_input)

    # Crop the transpose convolution output to match the shape of prev_layer_nonpool_input
    cropped_conv_transpose = tf.image.resize_with_crop_or_pad(
        conv_transpose,
        tf.shape(prev_layer_nonpool_input)[1],
        tf.shape(prev_layer_nonpool_input)[2],
    )

    merge = Concatenate(axis=3)([prev_layer_nonpool_input, cropped_conv_transpose])

    conv = Conv2D(n_filters, kernel_size=kernel_size, padding="same")(merge)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(n_filters, kernel_size=kernel_size, padding="same")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(n_filters, kernel_size=kernel_size, padding="same")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)

    return conv


def DecoderMiniBlock(
    prev_layer_input, prev_layer_nonpool_input, n_filters=32, kernel_size=(3, 3)
):
    """
    Build a mini-block for the decoder part of the U-Net.

    Parameters:
        prev_layer_input (Tensor): Input tensor to the previous layer.
        prev_layer_nonpool_input (Tensor): Input tensor to the previous layer without pooling.
        n_filters (int): Number of filters in the convolutional layers.
        kernel_size (tuple): Size of the convolutional kernel.

    Returns:
        Tensor: Output tensor of the mini-block.
    """
    # Upsampling and concatenation of feature maps
    up = UpSampling2D(size=(2, 2))(prev_layer_input)
    merge = Concatenate(axis=3)([prev_layer_nonpool_input, up])

    # Convolutional layers
    conv = Conv2D(n_filters, kernel_size=kernel_size, padding="same")(merge)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(n_filters, kernel_size=kernel_size, padding="same")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(n_filters, kernel_size=kernel_size, padding="same")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)

    return conv


def build_model(weights_path=None) -> tf.keras.Model:
    """
    Build the U-Net model using the specified architecture.

    Parameters:
        weights_path (str): Path to pre-trained weights.

    Returns:
        tf.keras.Model: Built U-Net model.
    """
    inputs = Input(shape=(None, None, 4))  # Input shape for images with 4 channels

    # Build the encoder part
    conv1_pool, conv1 = EncoderMiniBlock(inputs, n_filters=32)
    conv2_pool, conv2 = EncoderMiniBlock(conv1_pool, n_filters=64)
    conv3_pool, conv3 = EncoderMiniBlock(conv2_pool, n_filters=128)
    conv4_pool, conv4 = EncoderMiniBlock(conv3_pool, n_filters=256)
    conv5_pool, conv5 = EncoderMiniBlock(conv4_pool, n_filters=512)
    center = EncoderMiniBlock(
        conv5_pool, n_filters=1024, max_pooling=False, dropout_prob=0
    )

    # Build the decoder part
    deconv1 = DecoderMiniBlock(center, conv5, n_filters=512)
    deconv2 = DecoderMiniBlock(deconv1, conv4, n_filters=256)
    deconv3 = DecoderMiniBlock(deconv2, conv3, n_filters=128)
    deconv4 = DecoderMiniBlock(deconv3, conv2, n_filters=64)
    deconv5 = DecoderMiniBlock(deconv4, conv1, n_filters=32)

    # Output layer
    num_classes = 1
    outputs = Conv2D(
        num_classes, kernel_size=(1, 1), activation="sigmoid", dtype="float32"
    )(deconv5)

    # Create the U-Net model
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    # Load pre-trained weights (if provided)
    if weights_path is not None:
        unet_model.load_weights(weights_path)

    return unet_model


def load_unet_model(model_path):
    """
    Load a pre-trained U-Net model.

    Parameters:
        model_path (str): Path to the pre-trained model file.

    Returns:
        tf.keras.Model: The loaded U-Net model.
    """
    return tf.keras.models.load_model(model_path, compile=False)


def predict_cloud_mask(img_array, weights_path=None):
    """
    Predict cloud masks using a U-Net model for cloud masking.

    Parameters:
        img_array (numpy.ndarray): Input image array to predict the mask for.
        weights_path (str): Path to pre-trained weights.

    Returns:
        numpy.ndarray: Predicted cloud mask array.
    """
    # Normalize the input image array (adjust this as needed)
    img_array = img_array / 255.0

    # Calculate the amount of padding needed to make the dimensions divisible by 32
    pad_height = 32 - (img_array.shape[0] % 32)
    pad_width = 32 - (img_array.shape[1] % 32)

    # Pad the image array with mirror reflection
    padded_img_array = np.pad(
        img_array, ((0, pad_height), (0, pad_width), (0, 0)), mode="reflect"
    )

    # Create the U-Net model
    unet_model = build_model(weights_path)

    # Reshape the padded input array for prediction
    input_data = np.expand_dims(padded_img_array, axis=0)

    # Predict using the model
    prediction = unet_model.predict(input_data)

    # Remove the padded regions from the prediction
    prediction = prediction[:, : img_array.shape[0], : img_array.shape[1], :]

    # Apply thresholding to the prediction to generate the mask
    threshold = 0.5
    cloud_mask = np.where(prediction > threshold, 1, 0).astype(np.uint8)

    # Delete the model to free up memory
    del unet_model

    return cloud_mask[0]  # Get the mask from the batch dimension


def apply_cloud_mask(img_array, cloud_mask):
    """
    Apply a cloud mask to an image array.

    Parameters:
        img_array (numpy.ndarray): Input image array.
        cloud_mask (numpy.ndarray): Cloud mask to be applied.

    Returns:
        numpy.ndarray: Masked image array.
    """
    masked_img = np.copy(img_array)
    masked_img[
        cloud_mask[:, :, 0] == 1
    ] = 255  # Set cloud pixels to 255 in the first channel
    return masked_img


#####################################################################################################################
# Part 3: Applying the functions
#####################################################################################################################

# Using GPU operations

tf.test.is_built_with_cuda()

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Set memory growth to prevent GPU memory fragmentation
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        print("Using GPU:", gpus[0])

        # Verify the list of visible GPUs
        print("Visible GPUs:", tf.config.experimental.get_visible_devices("GPU"))
    except RuntimeError as e:
        print(e)

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the TIFF image
data_folder = (
    r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\planet_tiles"
)
# data_folder = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\planet_tiles\Test"
results_folder = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\planet_tiles\Processed Planet U-Net"

# Load the pre-trained U-Net model
model_path = (
    r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\cloud_model.h5"
)
# model_path = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\cloud_unet_tf2_00432_0_8636.h5"

# Create the U-Net model
# unet_model = load_unet_model(model_path)

# Print the model summary
# print(unet_model.summary())

# Get a list of Planet image files
planet_img_files = glob.glob(os.path.join(data_folder, "*.tif"))

# Initialize variables to store cloud fraction data and true labels
cloud_fraction_per_tile = []
true_labels = []

# Iterate through each Planet image and predict cloud masks
for planet_img_file in planet_img_files:
    with rasterio.open(planet_img_file) as src:
        img_array = src.read()  # Read image data
        img_array = np.moveaxis(img_array, 0, -1)  # Move channels dimension to the last

        cloud_mask = predict_cloud_mask(img_array, model_path)

        # Calculate cloud fraction for the current tile
        cloud_fraction = np.sum(cloud_mask) / cloud_mask.size
        cloud_fraction_per_tile.append(cloud_fraction)

        # Determine true label (0 if no clouds, 1 if clouds)
        true_label = 1 if cloud_fraction > 0.5 else 0
        true_labels.append(true_label)

        # Apply cloud mask to the image
        masked_img = apply_cloud_mask(img_array, cloud_mask)

        # Save the masked image
        base_filename = os.path.splitext(os.path.basename(planet_img_file))[0]
        output_path = os.path.join(results_folder, f"{base_filename}_masked.tif")

        # Create profile for output TIF with LZW compression
        profile = src.profile.copy()
        profile.update(
            {
                "dtype": "uint8",
                "count": masked_img.shape[0],  # Number of bands
                "height": masked_img.shape[1],  # Height
                "width": masked_img.shape[2],  # Width
                "compress": "lzw",  # LZW compression
            }
        )

        # Write the masked image using rasterio with LZW compression
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(masked_img)

#####################################################################################################################
# Part 4: Visualizing the results
#####################################################################################################################

# Visualize predictions and results

# Display the masked image
plt.imshow(masked_img)
plt.title("Masked Image")
plt.axis("off")
plt.show()

# Compute and visualize cloud_fraction_per_tile
plt.figure()
plt.plot(cloud_fraction_per_tile)
plt.xlabel("Tile Index")
plt.ylabel("Cloud Fraction")
plt.title("Cloud Fraction Per Tile")
plt.show()

# Visualize cloud fraction histograms
plt.figure()
plt.hist(cloud_fraction_per_tile, bins=20, edgecolor="black")
plt.xlabel("Cloud Fraction")
plt.ylabel("Frequency")
plt.title("Cloud Fraction Histogram")
plt.show()

# Plot the number of tiles that have clouds in each of the tiles
plt.figure()
plt.hist(
    np.array(cloud_fraction_per_tile) > 0.5,
    bins=2,
    edgecolor="black",
    align="mid",
    rwidth=0.8,
)
plt.xticks([0, 1], ["No Cloud", "Cloud"])
plt.xlabel("Cloud Presence")
plt.ylabel("Number of Tiles")
plt.title("Number of Tiles with Clouds vs. No Clouds")
plt.show()

# Compute predicted labels based on cloud fraction
predicted_labels = np.array(cloud_fraction_per_tile) > 0.5

# Compute and visualize confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Clouds", "Clouds"],
    yticklabels=["No Clouds", "Clouds"],
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# List all files in the results folder
masked_img_files = [
    os.path.join(results_folder, filename) for filename in os.listdir(results_folder)
]

# Select a random image file
random_masked_img_file = random.choice(masked_img_files)

# Open the random masked image using rasterio
with rasterio.open(random_masked_img_file) as src:
    masked_img_array = src.read()  # Read image data

    # Display the random masked image
    plt.imshow(masked_img_array)
    plt.title("Random Masked Image")
    plt.axis("off")
    plt.show()
