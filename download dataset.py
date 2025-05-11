import requests
import zipfile
import os

# URL of the dataset
url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"

# Path to save the dataset
zip_path = "EuroSAT.zip"

# Download the file
print("Downloading the dataset...")
response = requests.get(url)
with open(zip_path, "wb") as file:
    file.write(response.content)
print("Download complete!")

# Extract the file
print("Extracting the dataset...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("EuroSAT")
print("Extraction complete!")

# Clean up (optional)
os.remove(zip_path)
print("Dataset is ready in the 'EuroSAT' folder.")
