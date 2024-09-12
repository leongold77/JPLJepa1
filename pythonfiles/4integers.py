import pandas as pd

# Read the CSV file
input_file = 'csvs/1/videodata6.csv'
output_file = 'csvs/Full_dataset/modified_file6.csv'
df = pd.read_csv(input_file, header=None, names=['File Path', 'Class'], skiprows=1)

# Convert the second column to integers
df['Class'] = df['Class'].apply(lambda x: int(float(x)))

# Ensure the first column stays as a string
df['File Path'] = df['File Path']

# Write the modified DataFrame to a new CSV file
df.to_csv(output_file, header=False, index=False)

print(f"The modified CSV file has been created and saved as '{output_file}'.")