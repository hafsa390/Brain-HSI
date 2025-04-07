### Project title: Classification of hyperspectral images of brain using machine learning ### 
The goal of the project is to generate the pixelwise ground truths from hyperspectral images and utilize those ground truths for training different U-Net based models for semantic segmentation.

### Dataset description ### 

The publicly available [hyperspectral dataset of brain cancer](https://hsibraindatabase.iuma.ulpgc.es/) is used in this project.

This dataset contains hyperspectral image(HSI) cubes of brain tissue, ground truth images and RGB images. For each sample, only some of the pixels are labeled. 

| Tissue         | Number of labeled pixels |
|----------------|--------------------------|
| Tumor          |  42288                   |
| Normal         |  323151                  |
| Blood vessel   |  131704                  |
| Background     |  523574                  |
| Unlabeled      |  13157399                |

### Data preprocessing ###

The data preprocessing pipeline consists of below steps illustrated in data_preprocessing.py file:

- Calibration
- Moving average smoothing
- Extreme band reduction
- Optimal bands selection
- Spectral normalization

### Machine learning model training for pixelwise ground truth generation ###

The random forest model is used for pixelwise ground truth generation. 

Three-way data partition was performed with 5-fold cross validation. The data partition was done patient-wise. 
For each fold, the number of trees in RF model is optimized from 1 to 100 with a step size of 10. The macro F1 score is computed with different number of trees, and the model with best macro F1 score is considered for that fold.       

The preprocessing and generated classification map is shown in the below figure:
![Alt text]((https://docs.google.com/drawings/d/1PvT5kKrtu9c1SIFuCGWAXpQRAStot2cbPsmgJzWHVCc/edit?usp=sharing](https://github.com/hafsa390/Brain-HSI/blob/master/Untitled%20drawing.jpg))


