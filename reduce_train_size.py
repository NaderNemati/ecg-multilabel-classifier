import os
import pandas as pd
import random
import shutil

current_directory = os.getcwd()

# Path to the folder containing CSV files
from_directory = os.path.join('data', 'split_csvs', 'physionet_stratified_mini')

# Get the full path by joining the current working directory and the relative path
csv_root = os.path.join(current_directory, from_directory)

# List all CSV files in the folder that start with "train"
csv_files = [file for file in os.listdir(csv_root) if file.startswith('train') and file.endswith('.csv')]

# Desired reduction percentages
percentages = [0.5, 0.25, 0.125, 0.0675]

# Loop through each "train" CSV file
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    csv_path = os.path.join(csv_root, csv_file)
    df = pd.read_csv(csv_path)
    for p in percentages:    
        
        # Calculate the desired number of remaining rows (approximately 0.5, 0.25, 0.125, 0.0675)
        desired_remaining_rows = int(round(df.shape[0] * p))
        
        # Create a list of indices for the rows
        row_indices = list(range(df.shape[0]))
        
        # Randomly shuffle the indices
        random.shuffle(row_indices)
        
        # Delete rows from the shuffled list until desired remaining rows are reached
        for row_index in row_indices[desired_remaining_rows:]:
            df.drop(row_index, inplace=True)
        
        # Reset the DataFrame index
        df.reset_index(drop=True, inplace=True)
        
        # Create a subfolder inside the csv_root to save the modified CSV files
        new_folder = os.path.join(csv_root, f'physionet_stratified_mini_{p}')
        os.makedirs(new_folder, exist_ok=True)
        
        # Write the modified DataFrame back to a new CSV file inside the subfolder
        new_csv_file = os.path.join(new_folder, f"new_{csv_file}")
        df.to_csv(new_csv_file, index=False)

        # List all CSV files in the folder that start with "test"
        test_csv_files = [file for file in os.listdir(csv_root) if file.startswith('test') and file.endswith('.csv')]

        # Loop through each "test" CSV file
        for test_csv_file in test_csv_files:
            # Copy "test" CSV file to the new folder
            test_csv_path = os.path.join(csv_root, test_csv_file)
            shutil.copy(test_csv_path, new_folder)

print('Done!')



