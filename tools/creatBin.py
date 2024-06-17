import os

# Directory where the .bin files will be created
output_directory = "/data0/benke/ldx/openpcdet37/OpenPCDet/tools/final_result结果1/中间bin/1000/10个扰动点/Pedestrian/Moderate"

# Number of files to create
num_files = 7500

# Create the directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Generate and create the .bin files
for i in range(0, num_files + 1):
    file_path = os.path.join(output_directory, f"{i:06}.bin")
    with open(file_path, "wb"):
        pass

print(f"{num_files} empty .bin files created in {output_directory}.")