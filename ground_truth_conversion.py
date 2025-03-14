import os
import numpy as np
from spectral import envi

# this function will convert ground truth files (gtMap.hdr, gtMap) into numpy array .npy format
def convert_gt_to_npy(hsi_dir, base_dir, output_gt_dir):

    os.makedirs(output_gt_dir, exist_ok=True)

    # Loop through all files in the train directory
    for file in os.listdir(hsi_dir):
        if file.endswith('.npy'):
            # Extract the subject ID from the filename (e.g., '004-02' from '004-02.npy')
            subject_id = file.split('_128_bands.')[0]

            # Construct the paths for the corresponding ground truth files (gtMap.hdr and gtMap)
            gt_hdr_path = os.path.join(base_dir, subject_id, 'gtMap.hdr')
            gt_data_path = os.path.join(base_dir, subject_id, 'gtMap')

            if os.path.exists(gt_hdr_path) and os.path.exists(gt_data_path):
                # Read the ground truth data
                gt_map = envi.open(gt_hdr_path, gt_data_path).load()

                # Convert ground truth map to NumPy array
                gt_map_npy = np.array(gt_map)

                # Save the converted ground truth map as .npy file
                output_gt_path = os.path.join(output_gt_dir, f'{subject_id}-GT.npy')
                np.save(output_gt_path, gt_map_npy)

                print(f"Ground truth for {subject_id} saved to {output_gt_path}")
            else:
                print(f"Ground truth files for {subject_id} not found!")

# Path of directories
hsi_folder = "/media/hafsa/Hafsa/PhD_Projects/npj_database/preprocessed_HSI_data_npy"
base_directory = "/media/hafsa/Hafsa/PhD_Projects/npj_database/HSI_Human_Brain_Database_IEEE_Access (FirstCampaign)"
output_gt_directory = "/media/hafsa/Hafsa/PhD_Projects/npj_database/preprocessed_GTs_npy"

# Convert ground truth files
convert_gt_to_npy(hsi_folder, base_directory, output_gt_directory)
