import pandas as pd
import os 
import shutil

def combine_csv(csv_folder_path, output_folder):
    """
    Combines CSV files from dataset folders within the parent folder into a single CSV file.
    Modifies the first two columns by prefixing each entry with 'combined_data_setN'.

    Args:
        csv_folder_path (str): Path to the folder containing multiple dataset folders.
        output_folder (str): Path where the merged CSV will be stored.

    Returns:
        str: Path to the merged CSV file.
    """
    all_files = []

    for dataset in os.listdir(csv_folder_path):
        dataset_path = os.path.join(csv_folder_path, dataset)
        if os.path.isdir(dataset_path):  # Ensure it's a folder
            for file in os.listdir(dataset_path):
                if file.endswith(".csv"):  # Adjust for other formats like .tsv, .xlsx
                    file_path = os.path.join(dataset_path, file)
                    df = pd.read_csv(file_path)
                    
                    # Modify the first two columns by prefixing each entry
                    if len(df.columns) >= 2:
                        df[df.columns[0]] = df[df.columns[0]].apply(lambda x: f"{dataset}_{x}")
                        df[df.columns[1]] = df[df.columns[1]].apply(lambda x: f"{dataset}_{x}")
                    
                    all_files.append(df)

    # Combine all dataframes into one
    merged_df = pd.concat(all_files, ignore_index=True)

    # Save the combined dataset
    file_name = os.path.join(output_folder, "merged_dataset.csv")
    merged_df.to_csv(file_name, index=False)
    print(f"✅ All CSV files have been merged into: {file_name}")

    return file_name



def merge_image_datasets_wPattern(parent_folder, output_folder, target_subfolder_pattern):
    """
    Merges all images from dataset folders within the parent folder, but only from subfolders
    whose names match a specific pattern, into a single output folder. Renames images by prefixing 
    them with their dataset folder name to avoid filename conflicts.

    Args:
        parent_folder (str): Path to the folder containing multiple dataset folders.
        output_folder (str): Path where merged images will be stored.
        target_subfolder_pattern (str): Substring pattern to match subfolder names.

    Returns:
        None
    """
    # Create output folder if it doesn't exist
    out = output_folder+target_subfolder_pattern
    os.makedirs(out, exist_ok=True)

    # Loop through dataset folders inside the parent folder
    for dataset in os.listdir(parent_folder):
        dataset_path = os.path.join(parent_folder, dataset)

        if os.path.isdir(dataset_path):
            # Find subfolders that match the target pattern
            for subfolder in os.listdir(dataset_path):
                if target_subfolder_pattern in subfolder:
                    target_path = os.path.join(dataset_path, subfolder)
                    
                    if os.path.isdir(target_path):
                        for img_file in os.listdir(target_path):
                            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                                src_path = os.path.join(target_path, img_file)

                                # Rename file to prevent overwrites
                                new_filename = f"{dataset}_{img_file}"
                                dst_path = os.path.join(out, new_filename)

                                # Copy the image
                                shutil.copy(src_path, dst_path)

    print(f"✅ All images from subfolders containing '{target_subfolder_pattern}' have been merged into: {output_folder}")

    return output_folder+target_subfolder_pattern


