# Kalimantan Forest Height & Biomass Estimation Using Deep Learning

Welcome to the Kalimantan Forest Height & Biomass Estimation repository! This repository contains tools and models for estimating forest height and biomass using deep learning techniques, leveraging both LiDAR and Planet data.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [Data Pretreatment & Analysis](#data-pretreatment--analysis)
- [Modelling](#modelling)
- [Contributing](#contributing)
- [MIT License](#license)

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

[Provide instructions on how to get started with your project. This may include installation steps, dependencies, or any other essential information.]

## Data Pretreatment & Analysis

[Provide details about the data preprocessing and analysis steps. Explain the significance of these steps and how they contribute to the project's objectives.]

## Modelling

[Explain the modeling approach and the deep learning techniques used in your project. Highlight any specific models or algorithms.]

## Contributing

[Explain how others can contribute to your project. This may include information on submitting issues, pull requests, or guidelines for contributing code.]

## License

This project is licensed under the MIT License.

