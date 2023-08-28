# The objective of this file is to perform an analysis of the CHM files.

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

################################################################################################################################################################
# Part 2: Function definitions
################################################################################################################################################################


def plot_band_array(
    band_array, image_extent, title, cmap_title, colormap, colormap_limits
):
    """
    Function to plot a raster band.

    :param band_array: Array representing the raster band.
    :param image_extent: Extent of the image.
    :param title: Title of the plot.
    :param cmap_title: Title of the colormap.
    :param colormap: Colormap to be used.
    :param colormap_limits: Limits of the colormap.
    """
    plt.imshow(band_array, extent=image_extent)
    cbar = plt.colorbar()
    plt.set_cmap(colormap)
    plt.clim(colormap_limits)
    cbar.set_label(cmap_title, rotation=270, labelpad=20)
    plt.title(title)
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style="plain")
    rotatexlabels = plt.setp(ax.get_xticklabels(), rotation=90)


def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, epsg):
    """
    Function to convert a numpy array to a TIFF file.

    :param newRasterfn: File path of the new raster.
    :param rasterOrigin: Origin coordinates of the raster.
    :param pixelWidth: Width of each pixel.
    :param pixelHeight: Height of each pixel.
    :param array: Numpy array to be converted.
    :param epsg: EPSG code representing the coordinate reference system.
    """
    rows, cols = array.shape

    # Check if dimensions are valid
    if rows == 0 or cols == 0:
        raise ValueError("Invalid array dimensions for creating raster dataset")

    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName("GTiff")
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def raster2array(geotif_file):
    """
    Function to convert a raster to a numpy array.

    :param geotif_file: File path of the GeoTIFF file.
    :return: Numpy array representing the raster and metadata.
    """
    # sourcery skip: assign-if-exp, extract-method, merge-dict-assign, move-assign-in-block
    metadata = {}
    dataset = gdal.Open(geotif_file)
    metadata["array_rows"] = dataset.RasterYSize
    metadata["array_cols"] = dataset.RasterXSize
    metadata["bands"] = dataset.RasterCount
    metadata["driver"] = dataset.GetDriver().LongName
    metadata["projection"] = dataset.GetProjection()
    metadata["geotransform"] = dataset.GetGeoTransform()

    mapinfo = dataset.GetGeoTransform()
    metadata["pixelWidth"] = mapinfo[1]
    metadata["pixelHeight"] = mapinfo[5]

    metadata["ext_dict"] = {}
    metadata["ext_dict"]["xMin"] = mapinfo[0]
    metadata["ext_dict"]["xMax"] = mapinfo[0] + dataset.RasterXSize / mapinfo[1]
    metadata["ext_dict"]["yMin"] = mapinfo[3] + dataset.RasterYSize / mapinfo[5]
    metadata["ext_dict"]["yMax"] = mapinfo[3]

    metadata["extent"] = (
        metadata["ext_dict"]["xMin"],
        metadata["ext_dict"]["xMax"],
        metadata["ext_dict"]["yMin"],
        metadata["ext_dict"]["yMax"],
    )

    if metadata["bands"] == 1:
        raster = dataset.GetRasterBand(1)
        metadata["noDataValue"] = raster.GetNoDataValue()
        metadata["scaleFactor"] = raster.GetScale()

        # band statistics
        metadata["bandstats"] = {}
        stats = raster.GetStatistics(True, True)
        metadata["bandstats"]["min"] = round(stats[0], 2)
        metadata["bandstats"]["max"] = round(stats[1], 2)
        metadata["bandstats"]["mean"] = round(stats[2], 2)
        metadata["bandstats"]["stdev"] = round(stats[3], 2)

        array = (
            dataset.GetRasterBand(1)
            .ReadAsArray(0, 0, metadata["array_cols"], metadata["array_rows"])
            .astype(np.float32)  # Convert array to float32
        )
        array[array == int(metadata["noDataValue"])] = np.nan

        if metadata["scaleFactor"] is not None:
            array = array / metadata["scaleFactor"]
        else:
            array = array / 1.0

        return array, metadata

    else:
        print("More than one band... function only set up for single band data")


def crown_geometric_volume_pct(tree_data, min_tree_height, pct):
    """
    Function to calculate crown geometric volume percentiles.

    :param tree_data: Array of tree height data.
    :param min_tree_height: Minimum tree height.
    :param pct: Percentile value.
    :return: Crown geometric volume and the specified percentile.
    """
    p = np.percentile(tree_data, pct)
    tree_data_pct = [min(v, p) for v in tree_data]
    crown_geometric_volume_pct = np.sum(tree_data_pct - min_tree_height)
    return crown_geometric_volume_pct, p


def get_predictors(tree, chm_array, labels):
    """
    Function to get predictor variables from biomass data.

    :param tree: Tree object containing relevant data.
    :param chm_array: Array representing the canopy height model (CHM).
    :param labels: Array of labels indicating the tree indices.
    :return: List of predictor variables.
    """
    indexes_of_tree = np.asarray(np.where(labels == tree.label)).T
    tree_crown_heights = chm_array[indexes_of_tree[:, 0], indexes_of_tree[:, 1]]

    full_crown = np.sum(tree_crown_heights - np.min(tree_crown_heights))

    crown50, p50 = crown_geometric_volume_pct(
        tree_crown_heights, tree.min_intensity, 50
    )
    crown60, p60 = crown_geometric_volume_pct(
        tree_crown_heights, tree.min_intensity, 60
    )
    crown70, p70 = crown_geometric_volume_pct(
        tree_crown_heights, tree.min_intensity, 70
    )

    return [
        tree.label,
        np.float(tree.area),
        tree.major_axis_length,
        tree.max_intensity,
        tree.min_intensity,
        p50,
        p60,
        p70,
        full_crown,
        crown50,
        crown60,
        crown70,
    ]


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

    # Remove outliers based on different methods (you could also just put NaNs).
    # Think about this when you generalize to the full CHM dataset.
    tree_heights_filtered = tree_heights[
        (tree_heights >= lower_bound)
        & (tree_heights <= upper_bound)
        & (gradients <= threshold_gradient)
        & (z_scores <= threshold_z_score)
    ]

    plt.hist(tree_heights_filtered, bins=50, color="green", alpha=0.7)
    plt.xlabel("Tree Height (without outliers)")
    plt.ylabel("Frequency")
    plt.title("Tree Height Histogram (without outliers)")
    plt.show()

    return tree_heights_filtered


def process_lidar_files(lidar_folder_path, output_folder_path):
    """
    Function to process LiDAR files.

    :param lidar_folder_path: Path to the folder containing LiDAR files.
    :param output_folder_path: Path to the folder to save the processed files.
    """
    # Data Exploration
    lidar_tif_files = [
        file for file in os.listdir(lidar_folder_path) if file.endswith(".tif")
    ]

    # Data Preprocessing
    lidar_file_data = pd.DataFrame({"File": lidar_tif_files})
    lidar_file_data["Filepath"] = lidar_file_data["File"].apply(
        lambda x: os.path.join(lidar_folder_path, x)
    )

    lidar_file_data["Max Value"] = lidar_file_data["Filepath"].apply(
        get_max_tree_height
    )

    # Data Visualization - LiDAR (histogram)
    plt.hist(lidar_file_data["Max Value"], bins=20)
    plt.xlabel("Max Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Max Values in LiDAR TIFF Files")
    plt.show()

    # Outlier Detection - LiDAR (using different methods)
    lidar_feature_columns = ["Max Value"]
    X_lidar = lidar_file_data[lidar_feature_columns]

    scaler = StandardScaler()
    X_lidar_scaled = scaler.fit_transform(X_lidar)

    lidar_outlier_model = IsolationForest(contamination=0.05)
    lidar_file_data["Outlier"] = lidar_outlier_model.fit_predict(X_lidar_scaled)

    lidar_outliers = lidar_file_data[lidar_file_data["Outlier"] == -1]
    print(lidar_outliers)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Process LiDAR files
    for index, row in lidar_file_data.iterrows():
        file_path = row["Filepath"]
        output_file_path = os.path.join(output_folder_path, row["File"])
        tree_heights_filtered = generate_tree_height_histogram(file_path)

        dataset = gdal.Open(file_path)
        band = dataset.GetRasterBand(1)

        # Create output dataset
        output_driver = gdal.GetDriverByName("GTiff")
        output_dataset = output_driver.Create(
            output_file_path,
            band.XSize,
            band.YSize,
            1,
            band.DataType,
        )

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
            data_chunk = band.ReadAsArray(
                0, row_offset, band.XSize, row_end - row_offset
            )

            # Filter out non-tree values (e.g., ground, buildings) within the chunk
            data_chunk_filtered = np.where(data_chunk > 0, data_chunk, 0)

            # Write chunk of filtered data to the output dataset
            output_dataset.GetRasterBand(1).WriteArray(
                data_chunk_filtered, 0, row_offset
            )

        # Save and close the output dataset
        output_dataset.FlushCache()
        output_dataset = None

        # plt.savefig(output_file_path + "_histogram.png")
        plt.close()

    print("Processed files saved in the output folder.")


def load_chm_tiff(file_path):
    """
    Function to load a CHM TIFF file as a numpy array.

    :param file_path: File path of the CHM TIFF file.
    :return: Numpy array representing the CHM.
    """
    dataset = rasterio.open(file_path)
    chm_array = dataset.read(1)

    # Mask values less than 0 to np.nan
    chm_array[chm_array < 0] = np.nan

    return chm_array


def plot_histogram(data, xlabel, ylabel, title):
    """
    Function to plot a histogram.

    :param data: Data for the histogram.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param title: Title of the plot.
    """
    plt.hist(data, bins=50, color="blue", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def generate_chm_histogram(chm_array):
    """
    Function to generate a histogram for a CHM.

    :param chm_array: Array representing the CHM.
    """
    plot_histogram(
        chm_array.flatten(), "Height", "Frequency", "CHM Height Distribution"
    )


def plot_scatter(x, y, xlabel, ylabel, title):
    """
    Function to plot a scatter plot.

    :param x: Data for the x-axis.
    :param y: Data for the y-axis.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param title: Title of the plot.
    """
    plt.scatter(x, y, color="blue", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_box(data, xlabel, ylabel, title):
    """
    Function to plot a box plot.

    :param data: Data for the box plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param title: Title of the plot.
    """
    plt.boxplot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_heatmap(data, xlabels, ylabels, title):
    """
    Function to plot a heatmap.

    :param data: Data for the heatmap.
    :param xlabels: Labels for the x-axis.
    :param ylabels: Labels for the y-axis.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, cmap="viridis", xticklabels=xlabels, yticklabels=ylabels)
    plt.title(title)
    plt.show()


def calculate_statistics(data):
    """
    Function to calculate descriptive statistics.

    :param data: Data for which statistics are calculated.
    :return: Dictionary containing the calculated statistics.
    """
    stats = {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
    }
    return stats


def calculate_correlation_matrix(data):
    """
    Function to calculate a correlation matrix.

    :param data: Data for which the correlation matrix is calculated.
    :return: Correlation matrix.
    """
    corr_matrix = np.corrcoef(data.T)
    return corr_matrix


def perform_pca(data, n_components):
    """
    Function to perform Principal Component Analysis (PCA).

    :param data: Data for PCA.
    :param n_components: Number of principal components to retain.
    :return: Transformed data and explained variance ratio.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    return pca_data, explained_variance_ratio


def plot_interactive_scatter(data, x, y, labels):
    """
    Function to plot an interactive scatter plot with tooltips.

    :param data: Data for the scatter plot.
    :param x: Data for the x-axis.
    :param y: Data for the y-axis.
    :param labels: Labels for the data points.
    """
    fig = go.Figure(
        data=go.Scatter(
            x=data[x], y=data[y], mode="markers", text=labels, hoverinfo="text"
        )
    )
    fig.update_layout(xaxis_title=x, yaxis_title=y, title="Interactive Scatter Plot")
    fig.show()


def detect_outliers_zscore_chm(chm_array, threshold):
    """
    Function to detect outliers in a CHM using Z-scores.

    :param chm_array: Array representing the CHM.
    :param threshold: Z-score threshold for outlier detection.
    :return: Array of outliers.
    """
    z_scores = (chm_array - np.mean(chm_array)) / np.std(chm_array)
    outliers = chm_array[np.abs(z_scores) > threshold]
    return outliers


def detect_outliers_isolation_forest_chm(chm_array):
    """
    Function to detect outliers in a CHM using Isolation Forest.

    :param chm_array: Array representing the CHM.
    :return: Array ofoutliers.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(chm_array.reshape(-1, 1))
    clf = IsolationForest(random_state=0)
    clf.fit(scaled_data)
    outliers = chm_array[clf.predict(scaled_data) == -1]
    return outliers


def calculate_elevation_profile(chm_array, line_coordinates):
    """
    Function to calculate the elevation profile along a line.

    :param chm_array: Array representing the CHM.
    :param line_coordinates: List of (x, y) coordinates defining the line.
    :return: Distances along the line and corresponding elevation values.
    """
    # Extract elevation values along the line
    elevation_profile = []
    for coord in line_coordinates:
        elevation_profile.append(chm_array[int(coord[1]), int(coord[0])])

    # Calculate distance along the line
    distances = np.linspace(
        0,
        np.sqrt(
            (line_coordinates[-1][0] - line_coordinates[0][0]) ** 2
            + (line_coordinates[-1][1] - line_coordinates[0][1]) ** 2
        ),
        len(line_coordinates),
    )

    return distances, elevation_profile


def calculate_slope(chm_array, resolution):
    """
    This function calculates the slope of a Canopy Height Model (CHM) using gradient calculations.

    :param chm_array: Array representing the canopy height data.
    :param resolution: Resolution of the CHM (in meters per pixel).
    :return: Array representing the slope (in degrees).
    """
    # Calculate the gradient using np.gradient
    gradient_x, gradient_y = np.gradient(chm_array, resolution)

    # Calculate the slope in degrees using arctan of the gradient magnitude
    slope = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))

    # Convert the slope from radians to degrees
    slope = np.degrees(slope)

    return slope


def plot_canopy_height_vs_slope_histogram(chm_array, slope, num_bins=50):
    """
    Plot a 2D histogram of canopy height vs. slope.

    :param chm_array: Ground truth data for canopy height (numpy array).
    :param slope: Ground truth data for slope (numpy array).
    :param num_bins: Number of bins for both canopy height and slope (default: 50).
    """

    # Replace NaN values with zeros
    chm_array = np.nan_to_num(chm_array)
    slope = np.nan_to_num(slope)

    # Create the 2D histogram
    hist, bins_x, bins_y = np.histogram2d(
        chm_array.ravel(), slope.ravel(), bins=num_bins
    )

    # Plot the 2D histogram as a heatmap
    plt.imshow(
        hist.T,
        extent=[bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]],
        origin="lower",
        cmap="hot",
    )
    plt.colorbar(label="Number of Pixels")
    plt.xlabel("Canopy Height (meters)")
    plt.ylabel("Slope (degrees)")
    plt.title("Canopy Height vs. Slope 2D Histogram")
    plt.show()


def calculate_aspect(chm_array, resolution):
    """
    This function calculates the aspect of a Canopy Height Model (CHM) using gradient calculations.

    :param chm_array: Array representing the CHM.
    :param resolution: Resolution of the CHM.
    :return: Array representing the aspect.
    """
    # Calculate aspect using gradient
    dx, dy = np.gradient(chm_array, resolution)
    aspect = np.arctan2(-dy, dx)
    aspect = np.degrees(aspect)
    aspect[aspect < 0] += 360.0

    return aspect


def calculate_texture(chm_array, window_size):
    """
    Function to calculate texture measures of a CHM.

    :param chm_array: Array representing the CHM.
    :param window_size: Size of the sliding window for texture calculation.
    :return: Array representing the texture measures.
    """
    # Mask invalid pixels (NaN and negative values)
    mask = np.logical_or(np.isnan(chm_array), chm_array < 0)
    chm_array_masked = np.ma.masked_array(chm_array, mask=mask)

    # Calculate texture measures using sliding window on valid pixels
    texture = ndi.generic_filter(
        chm_array_masked, np.std, size=window_size, mode="constant", cval=np.NaN
    )

    return texture


def identify_gaps(chm_file, threshold_height=10, threshold_area=10, chunk_size=256):
    """
    Identify gaps in a LiDAR Canopy Height Model (CHM) TIFF file.

    :param chm_file: path to the CHM TIFF file
    :param threshold_height: threshold height to define gaps (default: 10)
    :param threshold_area: threshold area to define gaps (default: 10)
    :param chunk_size: size of the processing chunk (default: 256)
    :return: a binary mask indicating the gap areas (True for gaps, False for non-gaps)
    """
    with rasterio.open(chm_file) as src:
        # Get the CHM file size
        rows, cols = src.shape

        # Initialize the gap mask
        gap_mask = np.zeros((rows, cols), dtype=bool)

        for row in range(0, rows, chunk_size):
            for col in range(0, cols, chunk_size):
                # Read a chunk of the CHM data
                chm_data_chunk = src.read(
                    1, window=rasterio.windows.Window(col, row, chunk_size, chunk_size)
                )

                # Threshold the chunk to identify gap areas
                gap_mask_chunk = np.logical_and(
                    chm_data_chunk <= threshold_height, chm_data_chunk > 0
                )

                # Label the gap areas using connected components
                labeled_mask_chunk, num_labels = ndi.label(gap_mask_chunk)

                # Calculate the area of each labeled component
                component_sizes = np.bincount(labeled_mask_chunk.flatten())[1:]

                # Identify gaps based on area threshold
                gap_areas = np.where(component_sizes >= threshold_area)[0] + 1
                gap_mask_chunk = np.isin(labeled_mask_chunk, gap_areas)

                # Update the gap mask with the chunk results
                gap_mask[
                    row : row + chunk_size, col : col + chunk_size
                ] = gap_mask_chunk

    return gap_mask


def visualize_gaps(gap_mask):
    """
    Visualize the gap areas identified by the gap_mask.

    :param gap_mask: binary mask indicating the gap areas (True for gaps, False for non-gaps)
    """
    plt.imshow(gap_mask, cmap="hot")
    plt.title("Gap Areas")
    plt.colorbar(label="Gap")
    plt.show()


def compute_surface_area(folder_path):
    total_surface_pixels = 0
    pixel_size_meters = None

    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            filepath = os.path.join(folder_path, filename)
            with rasterio.open(filepath) as src:
                # Read the pixel size in meters from the metadata
                if not pixel_size_meters:
                    pixel_size_meters = abs(src.transform[0])

                # Count the total number of non-NaN pixels
                data = src.read(1, masked=True)
                total_surface_pixels += data.count()

    if pixel_size_meters is not None:
        # Calculate the total surface area in square meters
        total_surface_area_m2 = total_surface_pixels * (pixel_size_meters**2)

        # Convert total surface area from square meters to square kilometers
        total_surface_area_km2 = total_surface_area_m2 / 1e6

        return total_surface_pixels, total_surface_area_km2
    else:
        raise ValueError("No CHM TIFF files found in the folder.")


def check_number_of_bands(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            filepath = os.path.join(folder_path, filename)
            with rasterio.open(filepath) as src:
                num_bands = src.count
                print(f"Number of bands in {filename}: {num_bands}")


################################################################################################################################################################
# Part 3: Analysis
################################################################################################################################################################

# Define folder paths
lidar_folder_path = (
    r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\LiDAR"
)
output_folder_path = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\LiDAR\Processed LiDAR"
chm_file = r"C:\Users\mpetel\Documents\Kalimantan Project\Code\Data\Output\LiDAR\Processed LiDAR\Polygon_008_utm_50N.tif"

# Loading the CHM data
chm_array = load_chm_tiff(chm_file)

# Analysis the CHM data
total_pixels, total_area_km2 = compute_surface_area(lidar_folder_path)
print("Total surface covered by LiDAR data:")
print("In terms of pixels:", total_pixels)
print("In terms of square kilometers:", total_area_km2)

# Generate histogram for CHM
generate_chm_histogram(chm_array)

# Calculate and visualize slope
resolution = 1.0  # Define resolution (pixel size) of the CHM
slope = calculate_slope(chm_array, resolution)
plt.imshow(slope, cmap="hot")
plt.colorbar()
plt.title("Slope")
plt.show()

# Plot the 2D histogram of canopy height vs. slope
plot_canopy_height_vs_slope_histogram(chm_array, slope, num_bins=50)  # TO BE MODIFIED

# Calculate and visualize aspect
aspect = calculate_aspect(chm_array, resolution)
plt.imshow(aspect, cmap="terrain", vmin=0, vmax=360)
plt.colorbar()
plt.title("Aspect")
plt.show()

# Identifying and visualize gap masks
gap_mask = identify_gaps(chm_file)
visualize_gaps(gap_mask)

# Texture analysis
window_size = 100
texture = calculate_texture(chm_array, window_size)
plt.imshow(texture, cmap="gray")
plt.colorbar()
plt.title("Texture")
plt.show()
