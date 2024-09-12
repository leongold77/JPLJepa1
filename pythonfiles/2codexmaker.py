import pandas as pd

# Read the original CSV file
df = pd.read_csv(r'csvs/trains+outputs/train.csv')

# Get the unique strings in the leftmost column
unique_strings = df.iloc[:, 0].unique()

# Create a dictionary to map each unique string to an integer
string_to_int = {string: i for i, string in enumerate(unique_strings)}

# Create a new column with the integer values
df['Integer_Class'] = df.iloc[:, 0].map(string_to_int)

# Write the modified DataFrame to a new CSV file
df.to_csv(r'csvs/output.csv', index=False)

# Create a DataFrame for the codex
codex_df = pd.DataFrame(list(string_to_int.items()), columns=['String', 'Integer'])

# Write the codex DataFrame to a new CSV file
codex_df.to_csv(r'csvs/codex.csv', index=False)

print("The new column has been added and the modified CSV has been saved as 'output.csv'.")
print("The codex has been saved as 'codex.csv'.")

