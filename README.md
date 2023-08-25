# Kalimantan Forest Height & Biomass Estimation Using Deep Learning

Welcome to the Kalimantan Forest Height & Biomass Estimation repository! This repository contains tools and models for estimating forest height and biomass using deep learning techniques, leveraging both LiDAR and Planet data.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [Acknowledgments](#Acknowledgments)
- [License](#license)

## Project Overview

Forests are critical components of the global carbon cycle, acting as significant carbon reservoirs and absorbing a substantial amount of carbon dioxide. However, extensive forest loss and degradation have led to significant carbon emissions, affecting climate change goals. Protecting and restoring forests, especially in regions like Borneo and Kalimantan, are crucial for mitigating climate change and preserving biodiversity.

Two key forest properties, vegetation height, and canopy cover, play pivotal roles in forest management and monitoring. Accurate measurement of these properties is essential for understanding forest health, age, and carbon storage capacity. Deep learning and remote sensing technologies offer a promising approach to address these challenges.

This project focuses on the integration of LiDAR and Planet satellite data using deep learning techniques to model vegetation height and canopy cover in Kalimantan. By combining the precision of LiDAR with the broad coverage of satellite data, the project aims to create cost-effective and accurate forest vegetation maps. These maps have applications in land-cover classification, forest management, and monitoring processes like afforestation and deforestation.

The primary objectives of this study are:
- Develop deep learning models to estimate vegetation height and canopy cover.
- Integrate LiDAR and satellite data for precise and comprehensive forest mapping.
- Create continuous forest vegetation maps for forest management and monitoring.
- Provide a U-Net deep learning model for vegetation height prediction at a 100-meter spatial resolution.

By achieving these objectives, this project contributes to the conservation and sustainable management of Kalimantan's diverse and carbon-rich tropical forests.

## Folder Structure

The repository is structured into two main folders:

### Data Pretreatment & Analysis

This folder contains scripts and resources related to the preprocessing, treatment, and analysis of both LiDAR and Planet data. It includes:

- **CHM_Analysis**: Analysis of the CHM files.
- **CHM_Planet_Intersect**: Perform a matching file operation for CHM and Planet data.
- **CHM_Post-Processing**: Treat the CHM files.
- **Mapping_CHM_files**: Visualize the spatial distribution, height, and gradient of LiDAR data.

- **Planet_Analysis**: Analysis of the Planet files.
- **Planet_Post-Processing**: Create and apply cloud masks to Planet images through statistical methods. 
- **Planet_Post-Processing-U-Net**: Create and apply cloud masks to Planet images through U-Net modelling. 
- **Mapping_optical_tiles**: Visualize the spatial distribution of Planet data.

### Modelling

The Modelling folder is dedicated to deep learning model development. It includes the following subfiles:

- **Data_modelling**: Perform data preparation and modeling for random sampling.
- **Training_validation**: Train and validate deep learning models.
- **Predictions**: Perform model predictions at different levels.
- **Mapping_visualisation_NDVI_stratified_sampling**: Visualize the distribution of validation/training/test samples for NDVI stratified sampling.
- **Mapping_visualisation_random_sampling**: Visualize the distribution of validation/training/test samples for random sampling.

## Getting Started

### Prerequisites

Make sure you have the following prerequisites installed on your system:

- [Python](https://www.python.org/) (version `3.11.4`)
- [PyTorch](https://pytorch.org/) (version `11.7`)
- [Rasterio](https://rasterio.readthedocs.io/en/latest/)
- [GDAL](https://gdal.org/)

### Setup a virtual environment 
See details in the [venv documentation](https://docs.python.org/3/library/venv.html).
 
**Example on linux:**
 
Create a new virtual environment called `Kalimantan_Project_env`.
```
python3 -m venv /path/to/new/virtual/environment/Kalimantan_Project_env
```

Activate the new environment:
```
source /path/to/new/virtual/environment/Kalimantan_Project_env/bin/activate
```

### Install python packages
After activating the venv, install the python packages with:
```
pip install -r requirements.txt
```

## Acknowledgments

We extend our sincere appreciation to *Dr. Samuel Favrichon* for his invaluable guidance, expertise, and unwavering support throughout the course of this research. Dr. Favrichon's contributions in the realms of remote sensing, geospatial analysis, and forest ecology have been instrumental in shaping the direction and methodology of this study.

Our gratitude also extends to *Dr. Sassan Saatchi*, whose extensive knowledge of LiDAR technology and forest carbon monitoring provided critical insights that significantly enhanced the quality of our research.

This project would not have been possible without the collaboration and contributions of the Carbon Cycle and Ecosystems Group at *NASA Jet Propulsion Laboratory* (NASA-JPL). The collective expertise and dedication of this group have played a pivotal role in advancing our understanding of forest height and biomass estimation.

We acknowledge and appreciate the support of the *JPL Science* community for fostering an environment of innovation and collaboration, which has been instrumental in the success of this endeavor.

Lastly, we extend our thanks to the countless individuals and organizations who have contributed to the field of forest ecology and remote sensing, as their work has paved the way for advancements in forest monitoring and climate change mitigation.

## License

This project is licensed under the MIT License. See LICENSE for more details. 

