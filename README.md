### Project title: Classification of hyperspectral images of brain using machine learning ### 
The goal of the project is to generate the pixelwise ground truths from hyperspectral images and utilize those ground truths for training different U-Net based models for semantic segmentation.

### Dataset description ### 

The publicly available [hyperspectral dataset of brain cancer](https://hsibraindatabase.iuma.ulpgc.es/) is used in this project.

This dataset contains hyperspectral image(HSI) cubes of brain tissue, ground truth images and RGB images. For each sample, only some of the pixels are labeled. 

| Tissue         | Number of labeled pixels |
|----------------|--------------------------|
| Tumor          |  42288                   |
|----------------|--------------------------|
| Normal         |  323151                  |
|----------------|--------------------------|
| Blood vessel   |  131704                  |
|----------------|--------------------------|
| Background     |  523574                  |
|----------------|--------------------------|
| Unlabeled      |  13157399                |
|----------------|--------------------------|

### Data preprocessing ###

The data preprocessing pipeline consists of below steps illustrated in data_preprocessing.py file:

- Calibration
- Moving average smoothing
- Extreme band reduction
- Optimal bands selection
- Spectral normalization

### Machine learning model training for pixelwise ground truth generation ###



 


