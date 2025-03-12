# Project title: Classification of hyperspectral images of brain using machine learning # 
The goal of the project is to generate the pixelwise ground truths from hyperspectral images and utilize those ground truths for training different U-Net based models for semantic segmentation.

# Dataset description # 

The publicly available [hyperspectral dataset of brain cancer](https://hsibraindatabase.iuma.ulpgc.es/) is used in this project.

This dataset contains hyperspectral image(HSI) cubes of brain tissue, ground truth images and RGB images. For each sample, only some of the pixels are labeled. 

# Data preprocessing #

The data preprocessing pipeline consists of below steps illustrated in data_preprocessing.py file:

- Calibration
- Moving average smoothing
- Extreme band reduction
- Optimal bands selection
- Spectral normalization


 


