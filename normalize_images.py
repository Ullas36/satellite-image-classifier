import os
import cv2
import numpy as np
from tqdm import tqdm  # For progress tracking

# Path to the original dataset
dataset_path = r"D:\college\PROJECT\EuroSAT"

# Path to save normalized images
processed_path = r"D:\college\PROJECT\EuroSAT_Normalized"
os.makedirs(processed_path, exist_ok=True)

# List of categories (land cover types)
categories = os.listdir(dataset_path)

# Iterate through categories and subcategories
for category in categories:
    category_path = os.path.join(dataset_path, category)
    
    # Create a corresponding directory in the processed folder
    processed_category_path = os.path.join(processed_path, category)
    os.makedirs(processed_category_path, exist_ok=True)
    
    # Check for subdirectories
    subdirs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
    
    if not subdirs:
        print(f"No subdirectories found in {category_path}.")
        continue
    
    for subdir in subdirs:
        subdir_path = os.path.join(category_path, subdir)
        
        # Create a corresponding subdirectory in the processed folder
        processed_subdir_path = os.path.join(processed_category_path, subdir)
        os.makedirs(processed_subdir_path, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Normalize and save each image
        for image_file in tqdm(image_files, desc=f"Processing {subdir}", unit="image"):
            image_path = os.path.join(subdir_path, image_file)
            
            # Load the image
            image = cv2.imread(image_path)
            if image is not None:
                # Normalize the image
                normalized_image = image / 255.0  # Scale pixel values to [0, 1]
                
                # Convert normalized image back to 8-bit for saving (optional for consistency)
                normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)
                
                # Save the normalized image
                save_path = os.path.join(processed_subdir_path, image_file)
                cv2.imwrite(save_path, normalized_image_uint8)
            else:
                print(f"Failed to load image: {image_path}")

print("Normalization complete. Processed images saved to:", processed_path)
