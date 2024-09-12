import pandas as pd
import zipfile
import os

def filter_classes(input_file_path, output_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file_path)

    # Define the classes of interest
    classes_of_interest = {378: 0, 281: 1, 151: 2, 182: 3, 367: 4,
    103: 5, 104: 6, 77: 7, 155: 8, 264: 9,
    265: 10, 268: 11, 269: 12, 270: 13, 271: 14,
    272: 15, 273: 16, 274: 17, 275: 18, 90: 19,
    396: 20}

    # Filter the DataFrame to include only rows with the specified classes
    filtered_df = df[df['class'].isin(classes_of_interest)]

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file_path, index=False)

    return filtered_df


# Define file paths
input_file_path = 'csvs/Full_dataset/modified_file25.csv'
output_file_path = 'csvs/filtered_data/filter25.csv'
#zip_file_path = 'filtered_videos.zip'

# Run the function to filter classes
filtered_df = filter_classes(input_file_path, output_file_path)


