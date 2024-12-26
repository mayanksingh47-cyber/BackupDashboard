import os
import pandas as pd

# Specify the input and output directories
input_dir = r'C:\\python\\Raw_Data'  # Use a raw string or escape the backslashes
output_dir = r'C:\\python\\Cleaned_data'  # Use a raw string or escape the backslashes

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List of columns to keep
columns_to_keep = ['Customer', 'Server', 'Client', 'Job Type', 'Status', 'Backup Day']

# Iterate over each CSV file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Keep only the required columns
        df_filtered = df[columns_to_keep]
        
        # Remove rows where 'Job Type' is 'Restore'
        df_filtered = df_filtered[df_filtered['Job Type'] != 'Restore']
        
        # Convert 'Backup Day' to datetime format
        df_filtered['Backup Day'] = pd.to_datetime(df_filtered['Backup Day'], errors='coerce')

        # Create new columns for numerical features based on 'Backup Day'
        df_filtered['Backup Day Month'] = df_filtered['Backup Day'].dt.month
        df_filtered['Backup Day Day'] = df_filtered['Backup Day'].dt.day

        # Retain the original 'Backup Day' column alongside new columns

        # Save the filtered dataframe to the output directory with the same name
        output_path = os.path.join(output_dir, filename)
        df_filtered.to_csv(output_path, index=False)
        
        print(f"Processed file: {filename}")

print("All files processed successfully.")
