import os
import pandas as pd
import re

# Step 1: Create initial CSV from video file paths for each folder
def create_initial_csv(folder_paths, csv_folder_path):
    all_dfs = []
    for idx, folder_path in enumerate(folder_paths):
        # List all files in the folder
        file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

        # Extract the labels from the filenames (assuming the filenames are unique and meaningful)
        labels = [os.path.splitext(os.path.basename(file))[0] for file in file_list]

        # Create a DataFrame from the file list with extracted labels
        df = pd.DataFrame({'file_path': file_list, 'label': labels})

        # Save the DataFrame to a CSV file
        csv_path = os.path.join(csv_folder_path, f'videodata_{idx}.csv')
        df.to_csv(csv_path, index=False)
        all_dfs.append(df)
        print(f"Initial CSV file created for folder {folder_path} with {len(df)} entries.")

    return all_dfs

# Step 2: Merge initial CSV with another CSV based on common key
def merge_csv_files(initial_dfs, output_csv_path, merged_csv_folder_path):
    all_merged_dfs = []
    output_df = pd.read_csv(output_csv_path)

    for idx, initial_df in enumerate(initial_dfs):
        # Function to extract youtube_id safely
        def extract_youtube_id(file_path):
            match = re.search(r'([a-zA-Z0-9_-]{11})_', file_path)
            if match:
                return match.group(1)
            else:
                print(f"Warning: Could not extract youtube_id from file path: {file_path}")
                return None

        # Extract the youtube_id term from each file path
        initial_df['youtube_id'] = initial_df['file_path'].apply(extract_youtube_id)

        # Strip any leading or trailing whitespace from youtube_id columns
        initial_df['youtube_id'] = initial_df['youtube_id'].str.strip()
        output_df['youtube_id'] = output_df['youtube_id'].str.strip()

        # Merge the initial_df with output_df on the youtube_id
        merged_df = pd.merge(initial_df, output_df[['youtube_id', 'Integer_Class']], left_on='youtube_id', right_on='youtube_id', how='left')

        # Create a new DataFrame with the required format
        final_df = merged_df[['file_path', 'Integer_Class']]

        # Write the new DataFrame to a CSV file
        merged_csv_path = os.path.join(merged_csv_folder_path, f'merged_videodata_{idx}.csv')
        final_df.to_csv(merged_csv_path, header=False, index=False)
        all_merged_dfs.append(final_df)
        print(f"Merged CSV file created for folder {idx} with {len(final_df)} entries.")

    return all_merged_dfs

# Step 3: Convert the 'Integer_Class' column to integers
def convert_class_to_integer(merged_dfs, converted_csv_folder_path):
    all_converted_dfs = []
    for idx, merged_df in enumerate(merged_dfs):
        merged_df.columns = ['File Path', 'Class']
        merged_df['Class'] = merged_df['Class'].apply(lambda x: int(float(x)))
        converted_csv_path = os.path.join(converted_csv_folder_path, f'modified_file_{idx}.csv')
        merged_df.to_csv(converted_csv_path, header=False, index=False)
        all_converted_dfs.append(merged_df)
        print(f"The modified CSV file has been created and saved as '{converted_csv_path}' with {len(merged_df)} entries.")

    return all_converted_dfs

# Step 4: Filter the CSV files based on specific classes and map them to new integer values using the provided class mapping
def filter_classes(converted_dfs, class_mapping, final_filtered_csv_path):
    all_filtered_dfs = []

    for idx, converted_df in enumerate(converted_dfs):
        print(f"Filtering converted DataFrame {idx} with {len(converted_df)} entries.")
        filtered_df = converted_df[converted_df['Class'].isin(class_mapping.keys())]
        filtered_df['Class'] = filtered_df['Class'].map(class_mapping)
        all_filtered_dfs.append(filtered_df)
        print(f"Filtered DataFrame {idx} has {len(filtered_df)} entries.")

    # Combine all filtered dataframes into one
    mega_filtered_df = pd.concat(all_filtered_dfs, ignore_index=True)
    mega_filtered_df.to_csv(final_filtered_csv_path, index=False)
    print(f"The mega filtered CSV file has been created and saved as '{final_filtered_csv_path}' with {len(mega_filtered_df)} entries.")

# Step 5: Create a codex for the classes from 0 to 20 using the provided class mapping and save it to codex_output_csv_path
def create_codex(codex_csv_path, class_mapping, codex_output_csv_path):
    codex_df = pd.read_csv(codex_csv_path, header=None, names=['Class_Name', 'Original_Class'])
    codex_df['New_Class'] = codex_df['Original_Class'].map(class_mapping)
    codex_df = codex_df.dropna(subset=['New_Class'])
    codex_df.to_csv(codex_output_csv_path, index=False, columns=['Class_Name', 'Original_Class', 'New_Class'])
    print(f"The codex CSV file has been created and saved as '{codex_output_csv_path}'.")

# Define paths
folder_paths = [f"/home/leon-gold/jepa/kinetics-dataset/k400_targz/train/part_{i}" for i in range(60)]

csv_folder_path = r"/home/leon-gold/jepa/csvs/0/"
output_csv_path = r"csvs/trains+outputs/output.csv"
merged_csv_folder_path = r"csvs/1/"
converted_csv_folder_path = r"csvs/Full_dataset/"
final_filtered_csv_path = r"csvs/mega_filtered_data/mega_filtered2.csv"
codex_csv_path = r"csvs/codex.csv"
codex_output_csv_path = r"csvs/codex_output2.csv"

# Class mapping as specified
class_mapping = {
    103: 0, 104: 1, 151: 2, 182: 3, 122: 4,
    330: 5, 378: 6, 78: 7, 8: 8, 29: 9,
    74: 10, 82: 11, 83: 12, 131: 13, 132: 14,
    133: 15, 146: 16, 165: 17, 186: 18, 257: 19,
    261: 20, 164: 21
}

# Execute the steps
initial_dfs = create_initial_csv(folder_paths, csv_folder_path)
merged_dfs = merge_csv_files(initial_dfs, output_csv_path, merged_csv_folder_path)
converted_dfs = convert_class_to_integer(merged_dfs, converted_csv_folder_path)
filter_classes(converted_dfs, class_mapping, final_filtered_csv_path)
create_codex(codex_csv_path, class_mapping, codex_output_csv_path)
