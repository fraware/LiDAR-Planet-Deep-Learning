# The objective of this file is to perform basic analysis of the Planet files.

################################################################################################################################################################
## Part 1: Basic Analysis
################################################################################################################################################################

import math
import rasterio
import matplotlib.pyplot as plt
import numpy as np

image_file = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Polygon_014_utm_50S_merged.tif"
sat_data = rasterio.open(image_file)

### Calculating the dimensions of the image on earth in metres
width_in_projected_units = sat_data.bounds.right - sat_data.bounds.left
height_in_projected_units = sat_data.bounds.top - sat_data.bounds.bottom

print(f"Width: {width_in_projected_units}, Height: {height_in_projected_units}")

### Rows and Columns
print(f"Rows: {sat_data.height}, Columns: {sat_data.width}")

### Converting the pixel co-ordinates to longitudes and latitudes
# Upper left pixel
row_min = 0
col_min = 0

# Lower right pixel.  Rows and columns are zero indexing.
row_max = sat_data.height - 1
col_max = sat_data.width - 1

# Transform coordinates with the dataset's affine transformation.
topleft = sat_data.transform * (row_min, col_min)
botright = sat_data.transform * (row_max, col_max)

print(f"Top left corner coordinates: {topleft}")
print(f"Bottom right corner coordinates: {botright}")

### Bands
# The image that we are inspecting is a multispectral image consisting of 4 bands int he order B,G,R,N where N stands for near infrared.each band is stored as a numpy array.
print(f"Number of Bands: {sat_data.count}")

# sequence of band indexes
print(f"Sequence of Band Indexes: {sat_data.indexes}")

# Computing the total number of pixels in all Planet files
folder_path = (
    r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles"
)

# Initialize a variable to store the total pixel count
total_pixels = 0

# Iterate through all TIFF files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".tif") or filename.endswith(".tiff"):
        tiff_path = os.path.join(folder_path, filename)

        # Open the TIFF file using rasterio
        with rasterio.open(tiff_path) as src:
            # Get the number of pixels in the image (width x height)
            num_pixels = src.width * src.height

            # Add the number of pixels to the total count
            total_pixels += num_pixels

# Print the total number of pixels across all TIFF files
print(f"Total number of pixels in the folder: {total_pixels}")

## Visualising the Satellite Imagery

# We will use matplotlib to visualise the image since it essentially consists of arrays.

# Load the 4 bands into 2d arrays - recall that we previously learned PlanetScope band order is BGRN.
b, g, r, n = sat_data.read()

# Displaying the blue band.
fig = plt.imshow(b)
plt.show()

# Displaying the histogram for the blue band.
plt.hist(b.flatten(), bins=256, color="blue", alpha=0.7)
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("Blue Band Histogram")
plt.show()

# Displaying the green band.
fig = plt.imshow(g)
fig.set_cmap("gist_earth")
plt.show()

# Displaying the histogram for the green band.
plt.hist(g.flatten(), bins=256, color="green", alpha=0.7)
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("Green Band Histogram")
plt.show()

# Displaying the red band.
fig = plt.imshow(r)
fig.set_cmap("inferno")
plt.colorbar()
plt.show()

# Displaying the histogram for the red band.
plt.hist(r.flatten(), bins=256, color="red", alpha=0.7)
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("Red Band Histogram")
plt.show()

# Displaying the infrared band.
fig = plt.imshow(n)
fig.set_cmap("winter")
plt.colorbar()
plt.show()

# Displaying the histogram for the infrared band.
plt.hist(n.flatten(), bins=256, color="black", alpha=0.7)
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("Infrared Band Histogram")
plt.show()

# Create an RGB composite image
rgb_image = np.dstack((r, g, b))
plt.imshow(rgb_image)
plt.title("RGB Composite Image")
plt.show()

# Calculate color histograms
color_hist = []
for channel in range(3):
    hist, _ = np.histogram(rgb_image[:, :, channel], bins=256, range=[0, 256])
    color_hist.append(hist)

# Plot color histogram
colors = ["red", "green", "blue"]
plt.figure(figsize=(8, 5))
for channel in range(3):
    plt.plot(color_hist[channel], color=colors[channel], label=colors[channel])
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("Bands Histogram")
plt.legend()
plt.show()

# Display 4 bands plots at once

# Create a 5x1 subplot grid to display the bands.
fig, axs = plt.subplots(5, 1, figsize=(8, 12))

# Display the blue band in the first subplot.
axs[0].imshow(b, cmap="Blues")
axs[0].set_title("Blue Band")

# Display the green band in the second subplot.
axs[1].imshow(g, cmap="Greens")
axs[1].set_title("Green Band")

# Display the red band in the third subplot.
axs[2].imshow(r, cmap="Reds")
axs[2].set_title("Red Band")

# Display the infrared band in the fourth subplot.
axs[3].imshow(n, cmap="gray")
axs[3].set_title("Infrared Band")

# Create an RGB composite image and display in the fifth subplot.
rgb_image = np.dstack((r, g, b))
axs[4].imshow(rgb_image)
axs[4].set_title("RGB Composite")

# Adjust layout for better appearance and display the plot.
plt.tight_layout()
plt.show()

# Create a 5x1 subplot grid to display the bands' histograms.
fig, axs = plt.subplots(5, 1, figsize=(8, 12))

# Display the histogram for the blue band in the first subplot.
axs[0].hist(b.flatten(), bins=256, color="blue", alpha=0.7)
axs[0].set_xlabel("Pixel Value")
axs[0].set_ylabel("Frequency")
axs[0].set_title("Blue Band Histogram")

# Display the histogram for the green band in the second subplot.
axs[1].hist(g.flatten(), bins=256, color="green", alpha=0.7)
axs[1].set_xlabel("Pixel Value")
axs[1].set_ylabel("Frequency")
axs[1].set_title("Green Band Histogram")

# Display the histogram for the red band in the third subplot.
axs[2].hist(r.flatten(), bins=256, color="red", alpha=0.7)
axs[2].set_xlabel("Pixel Value")
axs[2].set_ylabel("Frequency")
axs[2].set_title("Red Band Histogram")

# Display the histogram for the infrared band in the fourth subplot.
axs[3].hist(n.flatten(), bins=256, color="black", alpha=0.7)
axs[3].set_xlabel("Pixel Value")
axs[3].set_ylabel("Frequency")
axs[3].set_title("Infrared Band Histogram")

# Create an RGB composite image and display in the fifth subplot.
rgb_image = np.dstack((r, g, b))
axs[4].imshow(rgb_image)
axs[4].set_title("RGB Composite")

# Adjust layout for better appearance and display the plot.
plt.tight_layout()
plt.show()


# Create a single subplot to display the histograms of all four bands.
fig, ax = plt.subplots(figsize=(10, 6))

# Display the histogram for the blue band.
ax.hist(b.flatten(), bins=256, color="blue", alpha=0.7, label="Blue")

# Display the histogram for the green band.
ax.hist(g.flatten(), bins=256, color="green", alpha=0.7, label="Green")

# Display the histogram for the red band.
ax.hist(r.flatten(), bins=256, color="red", alpha=0.7, label="Red")

# Display the histogram for the infrared band.
ax.hist(n.flatten(), bins=256, color="black", alpha=0.7, label="Infrared")

# Set labels and title
ax.set_xlabel("Pixel Value")
ax.set_ylabel("Frequency")
ax.set_title("Histograms of Satellite Bands")

# Add a legend to differentiate bands
ax.legend()

# Adjust layout for better appearance and display the plot.
plt.tight_layout()
plt.show()


################################################################################################################################################################
## Part 2: Generalization to other vegetation indices
################################################################################################################################################################

### Loading the necessary libraries
import rasterio
import numpy
import matplotlib.pyplot as plt
import os

# Load the red, green, and NIR bands from the raster file
filename = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\Output\planet_tiles\Polygon_014_utm_50S_merged.tif"
with rasterio.open(filename) as src:
    band_red = src.read(3)
    band_green = src.read(2)
    band_nir = src.read(4)
    band_blue = src.read(1)


def save_raster_with_metadata(output_filename, data, meta):
    # Set the source metadata as kwargs we'll use to write the new data:
    kwargs = meta.copy()

    # Update the 'dtype' value to match our array's dtype:
    kwargs.update(dtype=data.dtype)

    # Update the 'count' value since our output will be a single-band image:
    kwargs.update(count=1)

    # Write the new raster file with modified metadata:
    with rasterio.open(output_filename, "w", **kwargs) as dst:
        dst.write(data, 1)


# Define the MidpointNormalize class for color normalization
class MidpointNormalize(plt.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        plt.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# Enhance the visualize_raster function
def visualize_raster(data, title):
    # Set min/max values from data range for image
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # Set our custom midpoint for most effective analysis
    midpoint = 0.1

    # Set the colormap and normalize
    colormap = plt.cm.RdYlGn
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint)

    # Create a new figure
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    # Use 'imshow' to specify the input data, colormap, min, max, and norm for the colorbar
    cbar_plot = ax.imshow(data, cmap=colormap, norm=norm)

    # Turn off the display of axis labels
    ax.axis("off")

    # Set the title
    ax.set_title(title, fontsize=17, fontweight="bold")

    # Configure the colorbar
    cbar = fig.colorbar(cbar_plot, orientation="horizontal", shrink=0.65)

    # Show the plot
    plt.show()

    # Generate a histogram of the data values
    fig2 = plt.figure(figsize=(20, 10))
    ax = fig2.add_subplot(111)

    # Add a title & (x,y) labels to the plot
    plt.title(f"{title} Histogram", fontsize=18, fontweight="bold")
    plt.xlabel(f"{title} values", fontsize=14)
    plt.ylabel("Number of pixels", fontsize=14)

    # For the x-axis, we want to count every pixel that is not an empty value
    x = data[~np.isnan(data)]
    color = "g"

    # Call 'hist` with our x-axis, bins, and color details
    ax.hist(x, bins=50, color=color, histtype="bar", ec="black", range=(-1, 1))

    # Show the plot
    plt.show()


# NDVI
def calculate_ndvi(red_band, nir_band):
    ndvi = (nir_band.astype(float) - red_band.astype(float)) / (nir_band + red_band)
    return ndvi


# GSAVI (Green Soil Adjusted Vegetation Index)
def calculate_gsavi(green_band, nir_band):
    gsavi = (nir_band - green_band) / (nir_band + green_band + 0.5) * (1 + 0.5)
    return gsavi


# VARI (Visible Atmospherically Resistant Index)
def calculate_vari(red_band, green_band, blue_band):
    vari = (green_band.astype(float) - red_band.astype(float)) / (
        green_band + red_band - blue_band
    )
    return vari


# GNDVI (Green Normalized Vegetation Index)
def calculate_gndvi(green_band, nir_band):
    gndvi = (nir_band - green_band) / (nir_band + green_band)
    return gndvi


# CVI (Chlorophyll Vegetation Index)
def calculate_cvi(red_band, green_band, nir_band):
    cvi = nir_band * (green_band - red_band) / (green_band * (green_band + red_band))
    return cvi


# NDGI (Normalized Difference Greenness Index)
def calculate_ndgi(green_band, red_band):
    ndgi = (green_band - red_band) / (green_band + red_band)
    return ndgi


# GDVI (Green Difference Vegetation Index)
def calculate_gdvi(green_band, nir_band):
    gdvi = nir_band - green_band
    return gdvi


# MSAVI (Modified Soil Adjusted Vegetation Index)
def calculate_msavi(red_band, nir_band):
    msavi = (
        2 * nir_band + 1 - np.sqrt((2 * nir_band + 1) ** 2 - 8 * (nir_band - red_band))
    ) / 2
    return msavi


# DVI (Difference Vegetation Index)
def calculate_dvi(red_band, nir_band):
    dvi = nir_band - red_band
    return dvi


# SAVI (Soil Adjusted Vegetation Index)
def calculate_savi(red_band, green_band, nir_band):
    savi = (nir_band - red_band) / (nir_band + red_band + 0.5) * (1 + 0.5)
    return savi


# MSR (Modified Simple Ratio)
def calculate_msr(red_band, nir_band):
    msr = ((nir_band / red_band) - 1) / (np.sqrt(nir_band / red_band) + 1)
    return msr


# Choose the index you want to calculate (e.g., NDVI, GSAVI, GNDVI, etc.)
chosen_index = "GNDVI"

if chosen_index == "NDVI":
    index_result = calculate_ndvi(band_red, band_nir)
    index_title = "Normalized Difference Vegetation Index (NDVI)"
elif chosen_index == "GSAVI":
    index_result = calculate_gsavi(band_green, band_nir)
    index_title = "Green Soil Adjusted Vegetation Index (GSAVI)"
elif chosen_index == "VARI":
    index_result = calculate_vari(band_red, band_green, band_blue)
    index_title = "Visible Atmospherically Resistant Index (VARI)"
elif chosen_index == "GNDVI":
    index_result = calculate_gndvi(band_green, band_nir)
    index_title = "Green Normalized Vegetation Index (GNDVI)"
elif chosen_index == "CVI":
    index_result = calculate_cvi(band_red, band_green, band_nir)
    index_title = "Chlorophyll Vegetation Index (CVI)"
elif chosen_index == "NDGI":
    index_result = calculate_ndgi(band_green, band_red)
    index_title = "Normalized Difference Greenness Index (NDGI)"
elif chosen_index == "GDVI":
    index_result = calculate_gdvi(band_green, band_nir)
    index_title = "Green Difference Vegetation Index (GDVI)"
elif chosen_index == "MSAVI":
    index_result = calculate_msavi(band_red, band_nir)
    index_title = "Modified Soil Adjusted Vegetation Index (MSAVI)"
elif chosen_index == "DVI":
    index_result = calculate_dvi(band_red, band_nir)
    index_title = "Difference Vegetation Index (DVI)"
elif chosen_index == "SAVI":
    index_result = calculate_savi(band_red, band_green, band_nir)
    index_title = "Soil Adjusted Vegetation Index (SAVI)"
elif chosen_index == "MSR":
    index_result = calculate_msr(band_red, band_nir)
    index_title = "Modified Simple Ratio (MSR)"
else:
    print(f"Chosen index '{chosen_index}' is not supported.")

# Save the index raster
output_filename = (
    os.path.splitext(os.path.basename(filename))[0] + f"_{chosen_index.lower()}.tif"
)
save_raster_with_metadata(output_filename, index_result, src.meta)

# Visualize the index raster
visualize_raster(index_result, index_title)
