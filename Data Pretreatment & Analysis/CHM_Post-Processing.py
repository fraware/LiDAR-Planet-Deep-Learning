# The objective of this file is to treat the CHM files.

################################################################################################################################################################
# Part 1: Libraries implementation
################################################################################################################################################################

import os, sys
from osgeo import gdal, ogr, osr
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import pandas as pd
from skimage.segmentation import watershed
from skimage.morphology import local_minima
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import label
from shapely.geometry import shape
from rasterio.features import rasterize

################################################################################################################################################################
# Part 2: Function definitions
################################################################################################################################################################

def get_max_tree_height(chm_file):
    """
    Function to get the maximum tree height from a CHM TIFF file.

    :param chm_file: File path of the CHM TIFF file.
    :return: Maximum tree height.
    """
    try:
        with rasterio.open(chm_file) as src:
            chm_data = src.read(1)  # Read the first band of the CHM TIFF file

            # Filter out non-tree elements (e.g., ground, buildings) by considering only positive values
            tree_heights = chm_data[chm_data > 0]

            if len(tree_heights) > 0:
                max_tree_height = tree_heights.max()  # Compute the maximum tree height
                return max_tree_height
            else:
                print("No tree elements found in the CHM.")
                return None

    except FileNotFoundError:
        print("CHM TIFF file not found.")
        return None

def generate_tree_height_histogram(file_path):
    """
    Function to generate a histogram of tree heights from a CHM TIFF file.

    :param file_path: File path of the CHM TIFF file.
    :return: Filtered array of tree heights.
    """
    print("Generating tree height histogram for:", file_path)
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)

    # Define chunk size for processing in chunks
    chunk_size = 1000

    # Get raster dimensions
    width = band.XSize
    height = band.YSize

    tree_heights = []

    for row_offset in range(0, height, chunk_size):
        # Calculate row range for current chunk
        row_end = min(row_offset + chunk_size, height)

        # Read chunk of data
        data_chunk = band.ReadAsArray(0, row_offset, width, row_end - row_offset)

        # Filter out non-tree values (e.g., ground, buildings) within the chunk
        tree_heights_chunk = data_chunk[data_chunk > 0]

        # Append chunk to the overall tree heights
        tree_heights.extend(tree_heights_chunk)

    tree_heights = np.array(tree_heights)

    plt.hist(tree_heights, bins=50, color="blue", alpha=0.7)
    plt.xlabel("Tree Height")
    plt.ylabel("Frequency")
    plt.title("Tree Height Histogram")
    plt.show()

    return tree_heights


def explore_lidar_files(lidar_folder_path):
    """
    Explore LiDAR files in a folder.

    :param lidar_folder_path: Path to the folder containing LiDAR files.
    :return: DataFrame with file information.
    """
    lidar_tif_files = [file for file in os.listdir(lidar_folder_path) if file.endswith(".tif")]

    lidar_file_data = pd.DataFrame({"File": lidar_tif_files})
    lidar_file_data["Filepath"] = lidar_file_data["File"].apply(lambda x: os.path.join(lidar_folder_path, x))
    lidar_file_data["Max Value"] = lidar_file_data["Filepath"].apply(get_max_tree_height)

    return lidar_file_data


def visualize_lidar_files(lidar_file_data):
    """
    Visualize the distribution of maximum values in LiDAR TIFF files.

    :param lidar_file_data: DataFrame with file information.
    """
    print("Visualizing LiDAR files")
    plt.hist(lidar_file_data["Max Value"], bins=20)
    plt.xlabel("Max Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Max Values in LiDAR TIFF Files")
    plt.show()


def process_lidar_file(file_path, output_file_path):
    """
    Process a single LiDAR file.

    :param file_path: Path to the LiDAR TIFF file.
    :param output_file_path: Path to save the processed file.
    """
    tree_heights = generate_tree_height_histogram(file_path)
    tree_heights_filtered = remove_outliers(tree_heights)

    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)

    # Create output dataset
    output_driver = gdal.GetDriverByName("GTiff")
    output_dataset = output_driver.Create(output_file_path, band.XSize, band.YSize, 1, band.DataType)

    # Set the same geotransform, projection, and metadata as the input dataset
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())
    output_dataset.SetMetadata(dataset.GetMetadata())

    # Define chunk size for processing in chunks
    chunk_size = 1000

    for row_offset in range(0, band.YSize, chunk_size):
        # Calculate row range for current chunk
        row_end = min(row_offset + chunk_size, band.YSize)

        # Read chunk of data
        data_chunk = band.ReadAsArray(0, row_offset, band.XSize, row_end - row_offset)

        # Filter out non-tree values (e.g., ground, buildings) within the chunk
        data_chunk_filtered = np.where(data_chunk > 0, data_chunk, 0)

        # Write chunk of filtered data to the output dataset
        output_dataset.GetRasterBand(1).WriteArray(data_chunk_filtered, 0, row_offset)

    # Save and close the output dataset
    output_dataset.FlushCache()
    output_dataset = None


def process_lidar_files(lidar_folder_path, output_folder_path):
    """
    Function to process LiDAR files.

    :param lidar_folder_path: Path to the folder containing LiDAR files.
    :param output_folder_path: Path to the folder to save the processed files.
    """
    lidar_file_data = explore_lidar_files(lidar_folder_path)
    visualize_lidar_files(lidar_file_data)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Process LiDAR files
    for index, row in lidar_file_data.iterrows():
        file_path = row["Filepath"]
        output_file_path = os.path.join(output_folder_path, row["File"])
        print("Processing LiDAR file:", file_path)
        process_lidar_file(file_path, output_file_path)

    print("Processed files saved in the output folder.")


def remove_outliers(tree_heights):
    """
    Function to remove outliers from a given array of tree heights.

    :param tree_heights: Array of tree heights.
    :return: Filtered array of tree heights without outliers.
    """
    print("Removing outliers")
    # Outlier detection using different methods
    # Method 1: Quartiles
    q1 = np.percentile(tree_heights, 2.5)
    q3 = np.percentile(tree_heights, 97.5)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Method 2: Gradients
    gradients = np.gradient(tree_heights)
    mean_gradient = np.mean(gradients)
    std_gradient = np.std(gradients)
    threshold_gradient = mean_gradient + 2 * std_gradient

    # Method 3: Z-score
    z_scores = (tree_heights - np.mean(tree_heights)) / np.std(tree_heights)
    threshold_z_score = 2.5

    # Remove outliers based on different methods
    tree_heights_filtered = tree_heights[
        ((tree_heights >= lower_bound) & (tree_heights <= upper_bound))
        & (gradients <= threshold_gradient)
        & (z_scores <= threshold_z_score)
        | (tree_heights == -9999)
    ]

    plt.hist(tree_heights_filtered, bins=50, color="green", alpha=0.7)
    plt.xlabel("Tree Height (without outliers)")
    plt.ylabel("Frequency")
    plt.title("Tree Height Histogram (without outliers)")
    #plt.show()

    return tree_heights_filtered


def spot_invalid_pixels(chm_file, chunk_size, output_folder):
    """
    Procedure to spot and correct invalid pixel values in a CHM TIFF file.

    :param chm_file: path to the CHM TIFF file
    :param chunk_size: size of the chunk for processing
    :param output_folder: path to the folder where the corrected CHM TIFF will be saved
    :return: numpy array representing the cleaned CHM
    """
    # Open the CHM TIFF file in read mode
    with rasterio.open(chm_file, "r") as src:
        # Get the shape of the CHM data
        rows, cols = src.shape

        # Calculate the total number of pixels in the CHM data
        total_pixels = rows * cols

        # Initialize variables to keep track of the count of invalid pixels
        invalid_pixel_count = 0

        # Calculate the updated chunk size to fit within the raster dimensions
        chunk_size = min(chunk_size, rows, cols)

        # Create the output file path
        chm_file_name = os.path.splitext(os.path.basename(chm_file))[0]  # Get the filename without extension
        output_file = os.path.join(output_folder, f"{chm_file_name}_corrected.tif")

        # Open a new TIFF file for writing the corrected CHM
        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=rows,
            width=cols,
            count=1,
            dtype=src.meta["dtype"],
            crs=src.meta["crs"],
            transform=src.meta["transform"],
        ) as dst:
            # Iterate over chunks of the CHM data
            for row in range(0, rows, chunk_size):
                for col in range(0, cols, chunk_size):
                    # Calculate the valid window boundaries within the raster dimensions
                    window_row_start = row
                    window_row_end = min(row + chunk_size, rows)
                    window_col_start = col
                    window_col_end = min(col + chunk_size, cols)

                    # Read a chunk of the CHM data as a numpy array
                    chm_data = src.read(
                        1,
                        window=(
                            (window_row_start, window_row_end),
                            (window_col_start, window_col_end),
                        ),
                    )

                    # Identify invalid pixels (e.g., NoData values, negative values)
                    invalid_pixels = np.logical_or(
                        np.isnan(chm_data),
                        np.logical_or(chm_data == src.nodata, np.logical_or(chm_data < 0, chm_data == -9999)),
                    )

                    # Convert invalid pixel coordinates to global coordinates
                    global_invalid_pixels = (
                        invalid_pixels[0] + window_row_start,
                        invalid_pixels[1] + window_col_start,
                    )

                    # Increment the count of invalid pixels
                    invalid_pixel_count += len(global_invalid_pixels[0])

                    # Perform corrections for invalid pixels in the chunk
                    chm_data[invalid_pixels] = np.nan

                    # Write the corrected chunk of CHM data to the output file
                    dst.write(
                        chm_data,
                        1,
                        window=(
                            (window_row_start, window_row_end),
                            (window_col_start, window_col_end),
                        ),
                    )

        # Calculate the percentage of invalid pixels in the entire CHM data
        invalid_pixel_percentage = (invalid_pixel_count / total_pixels) * 100

        # Print the final percentage of invalid pixels
        print(f"Final percentage of invalid pixels: {invalid_pixel_percentage:.2f}%")

        # Print the number of invalid pixels
        print(f"Number of invalid pixels: {invalid_pixel_count}")

        # Return the path to the corrected CHM file
        return output_file


def determine_reasonable_border_width(chm_file, percentage=0.1, min_pixels=100):
    """
    Determine a reasonable border width in pixels for each side of the lidar polygon.

    :param chm_file: path to the CHM TIFF file containing the lidar polygon
    :param percentage: percentage of the bounding box dimensions to be used as border width (default: 0.1)
    :param min_pixels: minimum number of pixels to be used as border width (default: 100)
    :return: integer representing the border width in pixels
    """
    # Open the CHM TIFF file and read the CHM data
    with rasterio.open(chm_file) as src:
        chm = src.read(1)

    # Generate a binary mask from the CHM data (lidar polygon)
    lidar_polygon_mask = (chm != 0).astype(np.uint8)

    # Convert the binary mask into a polygon using rasterio.features.geometry_shapes
    polygons = rasterio.features.shapes(lidar_polygon_mask, transform=src.transform)
    lidar_polygon = shape(polygons.__next__()[0])  # Get the first polygon

    # Calculate the bounding box dimensions of the lidar polygon
    min_x, min_y, max_x, max_y = lidar_polygon.bounds

    # Determine a reasonable border width based on the bounding box dimensions
    width_pixels = max(max_x - min_x, max_y - min_y) * percentage
    border_width = max(int(width_pixels), min_pixels)

    # Print the pixel width
    print(f"pixel width: {width_pixels} pixels")

    # Print the border width
    print(f"border width: {border_width} pixels")

    return border_width


def remove_border_outliers(chm_file, threshold, output_folder, percentage=0.5, min_pixels=100):
    """
    Remove outliers from the border region of a CHM TIFF file.

    :param chm_file: path to the CHM TIFF file
    :param threshold: threshold value to identify outliers
    :param border_width: width of the border region (in pixels)
    :return: numpy array representing the CHM with outliers removed in the border region
    """
    # Open the CHM TIFF file
    with rasterio.open(chm_file) as src:
        chm = src.read(1)
        transform = src.transform

    # Calculate the total number of pixels
    total_pixels = chm.size

    # Determine a reasonable border width for each side of the lidar polygon
    border_region_width = determine_reasonable_border_width(chm_file, percentage=percentage, min_pixels=min_pixels)

    # Create a binary mask for the border region
    mask = np.zeros_like(chm, dtype=np.uint8)
    mask[border_region_width:-border_region_width, border_region_width:-border_region_width] = 1

    # Label connected components in the mask
    labeled_mask, num_labels = label(mask)

    # Calculate the tree heights in the border region
    border_heights = chm[labeled_mask == 1]

    print(border_heights)

    # Set outliers in the border as the tree heights greater than the threshold
    outliers = border_heights > threshold | (border_heights == -9999)
    chm_cleaned = chm.copy()
    chm_cleaned[labeled_mask == 1] = np.where(outliers, np.nan, chm[labeled_mask == 1])

    # Calculate and print the number of pixels that were removed
    removed_pixels = np.sum(outliers)

    # Calculate the total number of pixels in the border
    border_pixels = np.sum(labeled_mask == 1)

    # Print the total number of pixels
    print(f"Total number of pixels: {total_pixels}")

    # Print the number of pixels in the border
    print(f"Total number of pixels in the border: {border_pixels}")

    # Print the number of pixels that were removed
    print(f"Number of pixels removed: {removed_pixels}")

    # Save the cleaned CHM to the output folder with the "_cleaned" suffix
    chm_file_name = os.path.splitext(os.path.basename(chm_file))[0]  # Get the filename without extension
    output_file = os.path.join(output_folder, f"{chm_file_name}_cleaned.tif")

    with rasterio.open(chm_file) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(chm_cleaned, 1)

    # Print the path to the saved output file
    print(f"Cleaned CHM saved to: {output_file}")

    # Return the path to the corrected CHM file
    return output_file


def visualize_chm(chm_file, chm_cleaned_file, extent, title):
    """
    Visualize the original and cleaned CHM side by side.

    :param chm_file: path to the CHM TIFF file
    :param chm_cleaned: path to the cleaned CHM TIFF file
    :param extent: extent of the CHM data [min_x, max_x, min_y, max_y]
    :param title: title of the plot
    """
    # Read the original CHM
    with rasterio.open(chm_file) as src:
        chm = src.read(1)

    # Read the cleaned CHM
    with rasterio.open(chm_cleaned_file) as src:
        chm_cleaned_file = src.read(1)

    # Create a new figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original CHM
    axes[0].imshow(chm, extent=extent, cmap='terrain', vmin=0, vmax=20)
    axes[0].set_title("Original CHM")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # Plot the cleaned CHM
    axes[1].imshow(chm_cleaned_file, extent=extent, cmap='terrain', vmin=0, vmax=20)
    axes[1].set_title("CHM with Outliers Removed")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

################################################################################################################################################################
# Part 3: Outlier Removal Operations
################################################################################################################################################################

# 3 different outlier removal operations can be performed

# 1. Remove invalid tree heights and above threshold trees as outliers
# 2. Remove invalid pixels as outliers
# 3. Remove the outliers from the border region

# Define folder paths
lidar_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR"
output_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR"
chm_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\\Polygon_109_utm_50N.tif"
chm_processed_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\Polygon_109_utm_50N.tif"
chm_cleaned_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\Polygon_109_utm_50N_corrected.tif"
chm_corrected_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\LiDAR\Processed LiDAR\Polygon_109_utm_50N_corrected_cleaned.tif"

# Beginning of the operation
get_max_tree_height(chm_file)
generate_tree_height_histogram(chm_file)

# Process LiDAR files
process_lidar_files(lidar_folder_path, output_folder_path)


# Additional Techniques:
#########################

# Procedure to spot and correct invalid pixels values as outliers
chunk_size = 256
spot_invalid_pixels(chm_file, chunk_size, output_folder_path)

# check of the operation
get_max_tree_height(chm_corrected_file)
generate_tree_height_histogram(chm_corrected_file)

# Procedure to remove the outliers from the border region
# A smaller percentage value will result in a narrower border region.
# A larger percentage value will result in a wider border region.
# A smaller min_pixels value will ensure the border is not too narrow.
# A larger min_pixels value will set a minimum width for the border region.
threshold = 115  # Threshold value for outlier detection (e.g., 115 is the height of the tallest tree in the world)
remove_border_outliers(chm_corrected_file, threshold, output_folder_path, percentage=1, min_pixels=100)

# check of the operation
get_max_tree_height(chm_cleaned_file)
generate_tree_height_histogram(chm_cleaned_file)

# Get the extent of the CHM data
with rasterio.open(chm_file) as src:
    extent = src.bounds

# Visualize the original and cleaned CHM side by side
title = "Original vs. Cleaned Canopy Height Model (CHM)"
visualize_chm(chm_file, chm_cleaned_file, extent, title)
