import os

def check_and_update_file_list(txt_file_path):
    # Read the file paths from the txt file
    with open(txt_file_path, 'r') as file:
        file_paths = file.readlines()

    # Strip whitespace and newline characters
    file_paths = [path.strip() for path in file_paths]

    # Check for non-existent files
    existing_files = []
    for path in file_paths:
        if os.path.exists(path):
            existing_files.append(path)
        else:
            print(f"File does not exist: {path}")

    # Write back only the existing files to the txt file
    with open(txt_file_path, 'w') as file:
        for path in existing_files:
            file.write(path + '\n')

    print("File list has been updated. Non-existent file paths have been removed.")

# Example usage
txt_file_path = 'python_files/txt/walk_right.txt'  # Replace with the path to your txt file
check_and_update_file_list(txt_file_path)
