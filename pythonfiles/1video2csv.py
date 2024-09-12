import os
import pandas as pd

# Define the path to the folder containing the video data
folder_path = r"src/Sim_Vids/Big_Trial3_Class/Trial_Fixing"

# List all files in the folder
file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

# Extract the labels from the filenames (assuming the filenames are unique and meaningful)

# Create a DataFrame from the file list with extracted labels
df = pd.DataFrame({'file_path': file_list})

# Define the path to save the CSV file
csv_path = r"/home/leon-gold/jepa/csvs/0/0coolfix.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_path, index=False)

print("CSV file created successfully!")