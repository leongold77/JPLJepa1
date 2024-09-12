import os
import pandas as pd
import re

# Step 1: Create initial CSV from video file paths
def create_initial_csv(folder_path, csv_path):
    # List all files in the folder
    file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    # Extract the labels from the filenames (assuming the filenames are unique and meaningful)
    labels = [os.path.splitext(os.path.basename(file))[0] for file in file_list]

    # Create a DataFrame from the file list with extracted labels
    df = pd.DataFrame({'file_path': file_list, 'label': labels})

    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)
    print("Initial CSV file created successfully!")

# Step 2: Merge initial CSV with another CSV based on common key
def merge_csv_files(initial_csv_path, output_csv_path, merged_csv_path):
    file_paths_df = pd.read_csv(initial_csv_path)
    output_df = pd.read_csv(output_csv_path)

    # Function to extract youtube_id safely
    def extract_youtube_id(file_path):
        match = re.search(r'([a-zA-Z0-9_-]{11})_', file_path)
        if match:
            return match.group(1)
        else:
            print(f"Warning: Could not extract youtube_id from file path: {file_path}")
            return None

    # Extract the youtube_id term from each file path
    file_paths_df['youtube_id'] = file_paths_df['file_path'].apply(extract_youtube_id)

    # Strip any leading or trailing whitespace from youtube_id columns
    file_paths_df['youtube_id'] = file_paths_df['youtube_id'].str.strip()
    output_df['youtube_id'] = output_df['youtube_id'].str.strip()

    # Merge the file_paths_df with output_df on the youtube_id
    merged_df = pd.merge(file_paths_df, output_df[['youtube_id', 'Integer_Class']], left_on='youtube_id', right_on='youtube_id', how='left')

    # Create a new DataFrame with the required format
    final_df = merged_df[['file_path', 'Integer_Class']]

    # Write the new DataFrame to a CSV file
    final_df.to_csv(merged_csv_path, header=False, index=False)
    print("Merged CSV file created successfully!")

# Step 3: Convert the 'Integer_Class' column to integers
def convert_class_to_integer(input_file, output_file):
    df = pd.read_csv(input_file, header=None, names=['File Path', 'Class'], skiprows=0)
    df['Class'] = df['Class'].apply(lambda x: int(float(x)))
    df.to_csv(output_file, header=False, index=False)
    print(f"The modified CSV file has been created and saved as '{output_file}'.")

# Step 4: Filter the CSV file based on specific classes
def filter_classes(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path, header=None, names=['File Path', 'Class'])
    classes_of_interest = {378, 281, 151, 182, 367, 103, 104, 77, 155, 264, 265, 268, 269, 270, 271, 272, 273, 274, 275, 90, 396}
    filtered_df = df[df['Class'].isin(classes_of_interest)]
    filtered_df.to_csv(output_file_path, index=False)
    print(f"The filtered CSV file has been created and saved as '{output_file_path}'.")

# Define paths
folder_path = r"/home/leon-gold/jepa/kinetics-dataset/k400_targz/train/part_30"
initial_csv_path = r"/home/leon-gold/jepa/csvs/0/videodata30.csv"
output_csv_path = r"csvs/trains+outputs/output.csv"
merged_csv_path = r"csvs/1/videodata30.csv"
converted_csv_path = r"csvs/Full_dataset/modified_file30.csv"
filtered_csv_path = r"csvs/filtered_data/filter30.csv"

# Execute the steps
create_initial_csv(folder_path, initial_csv_path)
merge_csv_files(initial_csv_path, output_csv_path, merged_csv_path)
convert_class_to_integer(merged_csv_path, converted_csv_path)
filter_classes(converted_csv_path, filtered_csv_path)
