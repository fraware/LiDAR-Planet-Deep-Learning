# The purpose of this file is to perform a matching file operation for CHM and Planet data.

################################################################################################################################################################
# Part 1: Import necessary libraries
################################################################################################################################################################

import os
import glob
import re
import shutil
import geopandas as gpd
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.warp import transform_bounds
from rasterio.plot import show
from rasterio.warp import calculate_default_transform
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from shapely.ops import transform
from pyproj import Transformer
from pyproj import CRS
from pyproj import Proj, transform
from shapely.geometry import box
import numpy as np
import rasterio
import rasterio.windows
import tempfile
import rioxarray
import xarray
import uuid
import time
from scipy.ndimage import zoom

################################################################################################################################################################
# Part 2: Define Functions
################################################################################################################################################################

# Function to convert UTM zone to EPSG
def utm_zone_to_epsg(utm_zone):
    hemisphere = "north" if utm_zone[-1].upper() != "S" else "south"
    zone_number = int(utm_zone[:-1])
    epsg_code = 32600 + zone_number
    if hemisphere == "south":
        epsg_code += 100
    return epsg_code


# Function to delete temporary files
def delete_temp_files(file_list):
    for file_path in file_list:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting file: {file_path}, {e}")


# Function to convert a CRS to a desired one (this one works well)
def change_crs(input_file, output_file, new_crs):
    # Open the input optical file
    with rasterio.open(input_file) as src:
        # Retrieve the source CRS and metadata
        src_crs = src.crs
        kwargs = src.meta.copy()

        # Calculate the transform parameters for the reprojection
        transform, width, height = calculate_default_transform(
            src_crs, new_crs, src.width, src.height, *src.bounds
        )

        # Update the metadata to match the LiDAR resolution and new CRS
        kwargs.update(
            {"crs": new_crs, "transform": transform, "width": width, "height": height}
        )

        # Create the reprojected dataset
        with rasterio.open(output_file, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=new_crs,
                    resampling=Resampling.nearest,
                    dst_resolution=lidar_resolution,
                )

def create_dataset(data, crs, transform, chunk_size=128):
    # Reshape the data array if needed
    if data.ndim == 4:
        data = data[np.newaxis, :, :]

    # Create a temporary file to store the data
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
        output_file = tmpfile.name

        # Open a rasterio dataset to write the data to the temporary file
        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=data.shape[0],
            crs=crs,
            transform=transform,
            dtype="float32",
        ) as dataset:
            # Write the data in smaller chunks using rasterio.windows
            for i in range(0, data.shape[1], chunk_size):
                for j in range(0, data.shape[2], chunk_size):
                    window = rasterio.windows.Window(
                        j, i, min(chunk_size, data.shape[2] - j), min(chunk_size, data.shape[1] - i)
                    )
                    dataset.write(data[:, i : i + window.height, j : j + window.width], window=window)

    # Open the temporary file as a rasterio dataset and return it
    return rasterio.open(output_file)


################################################################################################################################################################
# Part 3: Apply the functions
################################################################################################################################################################

# Define the base folder path
# sourcery skip: use-named-expression
base_folder = (
    r"c:\Users\mpetel\Documents\Kalimantan Project\Code"
)

# Define the paths to the optical and LiDAR data folders
optical_folder = os.path.join(base_folder, "Data", "planet_tiles")
lidar_folder = os.path.join(base_folder, "Data", "LiDAR")
projection_shapefile = os.path.join(
    base_folder, "global_grid_planet_projlatlon", "global_grid_planet_projlatlon.shp"
)

# Define the output folder and subfolders
output_folder = os.path.join(base_folder, "Data", "Output")
lidar_output_folder = os.path.join(output_folder, "LiDAR")
optical_output_folder = os.path.join(output_folder, "planet_tiles")

# Create the output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(lidar_output_folder, exist_ok=True)
os.makedirs(optical_output_folder, exist_ok=True)

# Define the bounding box
bbox = box(108.00, -5.00, 120.00, 8.00)

# Load the projection shapefile and extract the bounding box
projection_data = gpd.read_file(projection_shapefile, bbox=bbox)
projection_bounds = projection_data.total_bounds
projection_box = box(
    projection_bounds[0],
    projection_bounds[1],
    projection_bounds[2],
    projection_bounds[3],
)

# Create a list of optical and LiDAR files
optical_files = glob.glob(os.path.join(optical_folder, "*.tif"))
lidar_files = glob.glob(os.path.join(lidar_folder, "*.tif"))

# Iterate over each LiDAR file
for lidar_file in lidar_files:
    # Extract the filename from the full file path
    lidar_filename = os.path.basename(lidar_file)

    # Open the LiDAR file
    with rasterio.open(lidar_file) as src:
        lidar_meta = src.meta
        lidar_bounds = src.bounds
        lidar_resolution = src.res[
            0
        ]

        # Extract UTM zone from the filename using regex
        utm_zone_match = re.search(r"_utm_(\d+[A-Z])", lidar_filename)

        if utm_zone_match:
            utm_zone = utm_zone_match[1]
            lidar_crs = utm_zone_to_epsg(utm_zone)

        # Create a bounding box for the LiDAR file
        lidar_box = box(
            *transform_bounds(
                lidar_crs,
                projection_data.crs,
                lidar_bounds.left,
                lidar_bounds.bottom,
                lidar_bounds.right,
                lidar_bounds.top,
            )
        )

        # Check if the LiDAR file intersects with the projection box
        if projection_data.intersects(lidar_box).any():
            # Create an empty list to store the cropped optical data
            cropped_optical_data = []

            lidar_files_intersection = projection_data[
                projection_data.intersects(lidar_box)
            ]

            # Extract the code from the lidar_files_intersection
            codes = lidar_files_intersection["name"].str.extract(r"(\d+-\d+)")[0]

            # Find the corresponding optical files with the extracted codes as partial matches
            matching_optical_files = []
            for code in codes:
                matching_files = glob.glob(
                    os.path.join(optical_folder, f"*{code}*.tif")
                )
                matching_optical_files.extend(matching_files)

            # Iterate over each optical file
            for optical_file in matching_optical_files:
                # Extract the filename from the full file path
                optical_filename = os.path.basename(optical_file)

                # Open the optical file
                with rasterio.open(optical_file) as src_optical:
                    # Change the CRS of the optical file
                    new_crs = "EPSG:4326"
                    new_optical_file = os.path.join(
                        optical_output_folder,
                        optical_filename.replace(".tif", "_crs_adjusted.tif"),
                    )

                    change_crs(optical_file, new_optical_file, new_crs)

                    # Open the reprojected optical data with rasterio
                    with rasterio.open(new_optical_file) as src_optical_reprojected:
                        optical_meta = src_optical_reprojected.meta
                        optical_bounds = src_optical_reprojected.bounds
                        optical_crs = src_optical_reprojected.crs

                        # Create a bounding box for the optical file
                        optical_box = box(
                            optical_bounds.left,
                            optical_bounds.bottom,
                            optical_bounds.right,
                            optical_bounds.top,
                        )

                        # Find the intersection of the two polygons
                        intersection = optical_box.intersection(lidar_box)

                        # Check if there is an intersection
                        if not intersection.is_empty:
                            print(
                                "Intersection found. Cropping and reprojecting the optical data..."
                            )

                            # Crop the optical data to the intersection area
                            crop_img, crop_transform = mask(
                                src_optical_reprojected,
                                shapes=[intersection],
                                crop=True,
                                # nodata=src_optical.nodata[0],  # Set the nodata value
                                all_touched=True,  # Include all pixels touched by the shape, not just fully covered ones
                                filled=True,  # Fill in any holes in the masked area
                            )

                            # Create a rasterio dataset for the cropped optical data
                            optical_dataset = create_dataset(
                                crop_img, optical_crs, crop_transform
                            )

                            cropped_optical_data.append(optical_dataset)

                            print("Cropping and reprojecting complete.")
                        else:
                            print("No intersection found. Skipping the optical file.")

            # Merge the cropped optical data
            print("Merging optical data...")
            merged_optical_data, merged_transform = merge(
                cropped_optical_data, resampling=Resampling.bilinear
            )
            print("Merging complete.")

            # Update the metadata of the merged optical data
            merged_optical_meta = optical_meta.copy()
            merged_optical_meta.update(
                {
                    "width": merged_optical_data.shape[2],
                    "height": merged_optical_data.shape[1],
                    "transform": merged_transform,
                    # "crs": lidar_crs
                }
            )

            # Construct the output file path for the merged optical data
            merged_optical_output_path = os.path.join(
                optical_output_folder, lidar_filename.replace(".tif", "_merged.tif")
            )

            # Write the merged optical data to a new file
            print(f"Saving merged optical data to file: {merged_optical_output_path}")
            with rasterio.open(
                merged_optical_output_path, "w", **merged_optical_meta
            ) as dst_optical:
                dst_optical.write(merged_optical_data)

            # Copy the LiDAR file to the output folder
            lidar_output_path = os.path.join(lidar_output_folder, lidar_filename)
            shutil.copy(lidar_file, lidar_output_path, follow_symlinks=True)

            print(f"Processed LiDAR file: {lidar_filename}")
            print(f"Merged optical file saved: {merged_optical_output_path}")
            print(f"Copied LiDAR file saved: {lidar_output_path}\n")
        else:
            print(f"No intersection found for LiDAR file: {lidar_filename}")

        # Delete the created files ending with "_crs_adjusted.tif" and "_reprojected.tif"
        crs_adjusted_files = glob.glob(os.path.join(optical_output_folder, "*_crs_adjusted.tif"))
        reprojected_files = glob.glob(os.path.join(optical_output_folder, "*_reprojected.tif"))
        delete_temp_files(crs_adjusted_files)
        delete_temp_files(reprojected_files)
