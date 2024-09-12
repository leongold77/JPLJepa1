import pandas as pd

def create_codex(input_codex_csv_path, output_codex_csv_path, class_mapping):
    # Read the input codex CSV file, skipping the first row (header)
    try:
        codex_df = pd.read_csv(input_codex_csv_path, header=0, names=['Class_Name', 'Original_Class'])
        print(f"Original codex:\n{codex_df}")
    except Exception as e:
        print(f"Error reading codex CSV: {e}")
        return

    # Ensure the Original_Class column is of type int
    try:
        codex_df = codex_df[1:]  # Skip the header row
        codex_df['Original_Class'] = codex_df['Original_Class'].astype(int)
        print(f"Codex after converting Original_Class to int:\n{codex_df}")
    except Exception as e:
        print(f"Error converting Original_Class to int: {e}")
        return

    # Map the Original_Class to the new class values using the provided class mapping
    try:
        codex_df['New_Class'] = codex_df['Original_Class'].map(class_mapping)
        print(f"Codex with new class mapping:\n{codex_df}")
    except Exception as e:
        print(f"Error mapping new class values: {e}")
        return

    # Drop rows where the New_Class is NaN (i.e., Original_Class not in the class_mapping)
    try:
        codex_df = codex_df.dropna(subset=['New_Class'])
        print(f"Codex after dropping unmapped classes:\n{codex_df}")
    except Exception as e:
        print(f"Error dropping unmapped classes: {e}")
        return

    # Save the new codex to the output CSV file
    try:
        codex_df.to_csv(output_codex_csv_path, index=False, columns=['Class_Name', 'Original_Class', 'New_Class'])
        print(f"The codex CSV file has been created and saved as '{output_codex_csv_path}'.")
    except Exception as e:
        print(f"Error saving codex CSV: {e}")

# Define paths
input_codex_csv_path = r"csvs/codex.csv"
output_codex_csv_path = r"csvs/codex_output.csv"

# Class mapping as specified
class_mapping = {
    378: 0, 281: 1, 151: 2, 182: 3, 367: 4,
    103: 5, 104: 6, 77: 7, 155: 8, 264: 9,
    265: 10, 268: 11, 269: 12, 270: 13, 271: 14,
    272: 15, 273: 16, 274: 17, 275: 18, 90: 19,
    396: 20
}

# Execute the codex creation
create_codex(input_codex_csv_path, output_codex_csv_path, class_mapping)
