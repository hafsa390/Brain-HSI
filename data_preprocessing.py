# Calibration
# Moving average filtering
# Extreme bands removals (Reducing 826 bands to 645 bands)
# Dimensionality reduction (Reducing 645 bands into N bands)
# Normalization of the spectral signatures to min and max values of 0 to 1
import gc
from spectral import envi
import numpy as np
from pysptools.noise import SavitzkyGolay
import os
import matplotlib.pyplot as plt
from scipy.io import savemat

base_path = "/media/hafsa/Hafsa/PhD_Projects/npj_database/ThirdCampaign"

for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)

    if not os.path.isdir(folder_path):
        continue

    print("Reading folder:", folder_name)

    # Create full paths using folder_name
    hdr_path = os.path.join(base_path, folder_name, "raw.hdr")
    raw_path = os.path.join(base_path, folder_name, "raw")
    white_ref_path = os.path.join(base_path, folder_name, "whiteReference.hdr")
    dark_ref_path = os.path.join(base_path, folder_name, "darkReference.hdr")

    # Load data
    h = envi.read_envi_header(hdr_path)
    wavelengths = [float(x) for x in h['wavelength']]
    raw_data = envi.open(hdr_path, raw_path)
    white_data = envi.open(white_ref_path, white_ref_path.replace(".hdr", ""))
    dark_data = envi.open(dark_ref_path, dark_ref_path.replace(".hdr", ""))

    data_nparr = np.array(raw_data.load())
    white_nparr = np.array(white_data.load())
    dark_nparr = np.array(dark_data.load())

    corrected_nparr_100 = 100 * np.divide(np.subtract(data_nparr, dark_nparr),
                                      np.subtract(white_nparr, dark_nparr))

    corrected_nparr = np.divide(np.subtract(data_nparr, dark_nparr),
                                          np.subtract(white_nparr, dark_nparr))

    print(np.min(corrected_nparr_100), np.max(corrected_nparr_100))
    print(np.min(corrected_nparr), np.max(corrected_nparr))

    print(corrected_nparr.shape)

    # moving average smoothing , like the nature's paper
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size   # [0.2, 0.2, 0.2, 0.2, 0.2]
    smoothed_cube = np.zeros_like(corrected_nparr)  # initialize an empty array to save the smoothed datacube

    # Apply the moving average filter along the spectral dimension of each cube for each pixel
    for i in range(corrected_nparr.shape[0]):  # Loop over height
        for j in range(corrected_nparr.shape[1]):  # Loop over width
            smoothed_cube[i, j, :] = np.convolve(corrected_nparr[i, j, :], kernel, mode='same') # Convolve the spectral data for each pixel


    print("Shape of the smoothed hyperspectral cube:", smoothed_cube.shape)   # Check the shape of the smoothed data

    reduced_cube = smoothed_cube[:, :, 55:700]   # the bands before 55th and 700th are discarded to avoid extreme bands into consideration
    print(reduced_cube.shape)
    reduced_wavelengths = wavelengths[55:700]
    number_of_bands = len(reduced_wavelengths)
    print(number_of_bands)

    lambda_max = max(reduced_wavelengths)
    lambda_min = min(reduced_wavelengths)

    desired_bands = 128
    sampling_interval = (lambda_max - lambda_min)/desired_bands  # sampling intervals for band selection is considered
    print(lambda_max, lambda_min)
    print(sampling_interval)

    selected_bands_indices = []
    current_wavelength = lambda_min

    # Find indices of wavelengths closest to the desired sampling points
    for _ in range(desired_bands):
        closest_index = np.argmin(np.abs(np.array(reduced_wavelengths)-current_wavelength)) # closest indices would be considered if indices have floating point values
        selected_bands_indices.append(closest_index)
        current_wavelength += sampling_interval

    # Get the 128 selected bands from the reduced cube
    selected_cube = reduced_cube[:, :, selected_bands_indices]
    # print(selected_cube)
    print(np.min(selected_cube), np.max(selected_cube))
    # Check the shape of the selected cube and selected wavelengths
    print("Shape of the selected hyperspectral cube (128 bands):", selected_cube.shape)
    selected_wavelengths = [reduced_wavelengths[i] for i in selected_bands_indices]
    print("Selected wavelengths:", selected_wavelengths)

    normalized_cube = np.zeros_like(selected_cube)

    for i in range(selected_cube.shape[0]):
        for j in range(selected_cube.shape[1]):
            spectral_signature = selected_cube[i, j, :]
            min_val = np.min(spectral_signature)
            max_val = np.max(spectral_signature)

            # normaliza the spectral signature
            if max_val > min_val: # Avoid division by zero
                normalized_cube[i, j, :] = (spectral_signature - min_val)/(max_val-min_val)
            else:
                normalized_cube[i, j, :] = 0.0

    print("Shape of the normalized HSI cube: ", normalized_cube.shape)

    output_dir = "/media/hafsa/Hafsa/PhD_Projects/npj_database/preprocessed_HSI_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    npy_file_path = os.path.join(output_dir, f'{folder_name}_128_bands.npy')
    np.save(npy_file_path, normalized_cube)

    print(f"Saved {folder_name}_128_bands.npy, Shape: {normalized_cube.shape}")

    del raw_data, white_data, dark_data, corrected_nparr, smoothed_cube, reduced_cube, selected_cube, normalized_cube
    gc.collect() # garbage collection



    # vis_cube = normalized_cube[:, :, 125]
    # # Right: Smoothed image
    # plt.subplot(1, 2, 2)
    # plt.imshow(vis_cube, cmap='gray')
    # #plt.title(f'Smoothed Band {band_number + 1}')
    # plt.colorbar(label='Intensity')
    # # Display the plot
    # plt.tight_layout()
    # plt.show()

    # band_number = 560
    # # Plot both images side by side
    # plt.figure(figsize=(12, 6))
    #
    # unsmoothed_cube = corrected_nparr[:,:,band_number]
    # print(np.min(unsmoothed_cube), np.max(unsmoothed_cube))
    # smoothed_cube = smoothed_cube[:,:,band_number]
    # print(np.min(smoothed_cube), np.max(smoothed_cube))
    # print((unsmoothed_cube==smoothed_cube).all())
    # # Left: Original image
    # plt.subplot(1, 2, 1)
    # plt.imshow(unsmoothed_cube, cmap='gray')
    # plt.title(f'Original Band {band_number + 1}')
    # plt.colorbar(label='Intensity')
    #
    # # Right: Smoothed image
    # plt.subplot(1, 2, 2)
    # plt.imshow(smoothed_cube, cmap='gray')
    # plt.title(f'Smoothed Band {band_number + 1}')
    # plt.colorbar(label='Intensity')
    #
    # # Display the plot
    # plt.tight_layout()
    # plt.show()
