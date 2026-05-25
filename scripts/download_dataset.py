import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("miguelcorraljr/ted-ultimate-dataset")

# Copy the dataset to the current directory
destination = os.path.join(os.getcwd(), "ted-ultimate-dataset")
if not os.path.exists(destination):
    shutil.copytree(path, destination)

print("Path to dataset files:", destination)