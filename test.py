import os

# Specify the path to the CSV file
csv_file_path = 'Data/test_data2.csv'

# Check if the file exists
if os.path.exists(csv_file_path):
    print(f"File found: {csv_file_path}")
else:
    print(f"File not found: {csv_file_path}")