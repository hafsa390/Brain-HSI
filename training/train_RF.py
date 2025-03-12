import os
import glob
import shutil
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import joblib

# Define the paths of directories
data_path = "/media/hafsa/Hafsa/PhD_Projects/npj_database/preprocessed_HSI_data_npy"
ground_truth_path = "/media/hafsa/Hafsa/PhD_Projects/npj_database/preprocessed_GTs_npy"
output_base_path = "/media/hafsa/Hafsa/PhD_Projects/npj_database/5_Fold_CV_RF"

# Create output base directory
os.makedirs(output_base_path, exist_ok=True)

# All HSI cubes and ground truth files are sorted
hsi_cube_files = sorted(glob.glob(os.path.join(data_path, '*.npy')))
ground_truth_files = sorted(glob.glob(os.path.join(ground_truth_path, '*.npy')))

assert len(hsi_cube_files) == len(ground_truth_files), "Mismatch in the number of cubes and ground truth files."

# Group files by patient identifier
patient_data = {}
for hsi_file, gt_file in zip(hsi_cube_files, ground_truth_files):
    patient_id = os.path.basename(hsi_file).split('-')[0]
    if patient_id not in patient_data:
        patient_data[patient_id] = {'hsi': [], 'gt': []}
    patient_data[patient_id]['hsi'].append(hsi_file)
    patient_data[patient_id]['gt'].append(gt_file)

# Get all patient IDs
patient_ids = list(patient_data.keys())  # all the patient ids based on which the data will be split

# Helper function to save files
def save_files(patient_ids, target_path):
    for patient_id in patient_ids:
        hsi_files = patient_data[patient_id]['hsi']
        gt_files = patient_data[patient_id]['gt']
        for hsi_file, gt_file in zip(hsi_files, gt_files):
            shutil.copy(hsi_file, os.path.join(target_path, "data", os.path.basename(hsi_file)))
            shutil.copy(gt_file, os.path.join(target_path, "GT", os.path.basename(gt_file)))


 # Prepare data
def prepare_data(hsi_files_path, gt_files_path):
    hsi_cube_files = sorted(glob.glob(os.path.join(hsi_files_path, '*.npy')))
    ground_truth_files = sorted(glob.glob(os.path.join(gt_files_path, '*.npy')))
    labeled_pixels = []
    labeled_labels = []
    for hsi_file, gt_file in zip(hsi_cube_files, ground_truth_files):
        hsi_cube = np.load(hsi_file)
        ground_truth = np.load(gt_file)
        flat_hsi = hsi_cube.reshape(-1, hsi_cube.shape[2])
        flat_gt = ground_truth.flatten()
        mask = flat_gt != 0
        labeled_pixels.append(flat_hsi[mask])
        labeled_labels.append(flat_gt[mask])
    labeled_pixels1 = np.vstack(labeled_pixels)
    labeled_labels1 = np.concatenate(labeled_labels)
    return labeled_pixels1, labeled_labels1


# KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_val_ids, test_ids in kf.split(patient_ids):
    print(f"Processing Fold {fold}...")

    # Split patient IDs into train+val and test
    train_val_ids = [patient_ids[i] for i in train_val_ids]
    test_ids = [patient_ids[i] for i in test_ids]

    # Further split train+val into train and validation
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.25, random_state=42)

    # Create fold-specific directories
    fold_path = os.path.join(output_base_path, f"Fold_{fold}")
    train_path = os.path.join(fold_path, "train")
    val_path = os.path.join(fold_path, "val")
    test_path = os.path.join(fold_path, "test")
    for folder in [train_path, val_path, test_path]:
        os.makedirs(os.path.join(folder, "data"), exist_ok=True)
        os.makedirs(os.path.join(folder, "GT"), exist_ok=True)


    # Save data for the current fold
    save_files(train_ids, train_path)
    save_files(val_ids, val_path)
    save_files(test_ids, test_path)


    X_train, y_train = prepare_data(os.path.join(train_path, "data"), os.path.join(train_path, "GT"))
    X_val, y_val = prepare_data(os.path.join(val_path, "data"), os.path.join(val_path, "GT"))
    X_test, y_test = prepare_data(os.path.join(test_path, "data"), os.path.join(test_path, "GT"))

    # Hyperparameter tuning for kNN
    best_macro_f1 = -1
    best_n_trees = None
    macro_f1_scores = []

    print(f"\nFold {fold} - Macro F1-Scores for different number of tree values:")
    print("=" * 60)

    for n_trees in range(1, 101, 10):  # Number of trees from 1 to 100 with step size 10
        rf_model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf_model.fit(X_train, y_train)
        y_val_pred = rf_model.predict(X_val)

        # Filter out background class (label 4) for Macro F1-Score computation
        non_bg_indices = y_val != 4
        macro_f1 = f1_score(y_val[non_bg_indices], y_val_pred[non_bg_indices], average='macro')
        macro_f1_scores.append((n_trees, macro_f1))

        print(f"Number of trees: {n_trees} -> Macro F1-Score (excluding background): {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_n_trees = n_trees
            best_rf_model = rf_model

    print("=" * 60)
    print("\nSummary of Macro F1-Scores:")
    for n_trees, macro_f1 in macro_f1_scores:
        print(f"Number of trees: {n_trees}, Macro F1-Score: {macro_f1:.4f}")

    # Save the best model for this fold
    model_save_path = os.path.join(test_path, f"Best_RF_Model_Fold_{fold}.pkl")
    joblib.dump(best_rf_model, model_save_path)
    print(f"\nBest RF model for Fold {fold} saved to: {model_save_path}")
    print(f"Best number of trees: {best_n_trees}, Best Macro F1-Score: {best_macro_f1:.4f}")
    print(f"Fold {fold} completed")
    fold += 1

print("\n5-Fold Cross-Validation Completed.")
