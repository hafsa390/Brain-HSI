Project title: Classification of hyperspectral images of brain using machine learning 

#Dataset description# 

The publicly available [hyperspectral dataset of brain cancer](https://hsibraindatabase.iuma.ulpgc.es/) is used in this project.

This dataset contains hyperspectral image(HSI) cubes of brain tissue, ground truth images and RGB images. For each sample, only some of the pixels are labeled. 

# Data preprocessing #

The data preprocessing pipeline consists of below steps illustrated in data_preprocessing.py file:

- Calibration
- Moving average smoothing
- Extreme band reduction
- Optimal bands selection
- Spectral normalization


 


