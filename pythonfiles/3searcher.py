import pandas as pd
import re

# Read the CSV file with file paths and placeholders
file_paths_df = pd.read_csv(r'csvs/0/videodata6.csv', header=None, names=['file_path', 'placeholder'])

# Read the output_new.csv file
output_df = pd.read_csv(r'csvs/trains+outputs/output.csv')



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

# Check for any NaN values in the extracted youtube_id
if file_paths_df['youtube_id'].isnull().any():
    print("Warning: Some youtube_id values could not be extracted.")
    print(file_paths_df[file_paths_df['youtube_id'].isnull()])

# Strip any leading or trailing whitespace from youtube_id columns
file_paths_df['youtube_id'] = file_paths_df['youtube_id'].str.strip()
output_df['youtube_id'] = output_df['youtube_id'].str.strip()

# Merge the file_paths_df with output_df on the youtube_id
merged_df = pd.merge(file_paths_df, output_df[['youtube_id', 'Integer_Class']], left_on='youtube_id', right_on='youtube_id', how='left')

# Create a new DataFrame with the required format
final_df = merged_df[['file_path', 'Integer_Class']]

# Write the new DataFrame to a CSV file
final_df.to_csv('csvs/1/videodata6.csv', header=False, index=False)

print("The new CSV file has been created and saved as 'updated_file_paths_with_classes.csv'.")


