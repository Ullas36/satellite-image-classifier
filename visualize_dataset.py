import os
import matplotlib.pyplot as plt
import cv2

# Path to the dataset (update with your folder path)
dataset_path = r"D:\college\PROJECT\EuroSAT"  # Raw string path to avoid escape sequences

# List of categories (land cover types)
categories = os.listdir(dataset_path)

# Display sample images from each subfolder (limit to the first 5 subfolders for simplicity)
plt.figure(figsize=(15, 10))

count = 1  # To keep track of the subplot index
for i, category in enumerate(categories):  # Iterate through all categories
    category_path = os.path.join(dataset_path, category)
    
    # Check for subdirectories
    subdirs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
    
    if not subdirs:
        print(f"No subdirectories found in {category_path}.")
        continue
    
    for subdir in subdirs:  # Loop through subdirectories
        subdir_path = os.path.join(category_path, subdir)
        
        # Get all image files (jpg, png, jpeg) from the subdirectory
        image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if image_files:
            for idx in range(min(2, len(image_files))):  # Show first 2 images or less if there are fewer than 2
                sample_image = image_files[idx]
                image_path = os.path.join(subdir_path, sample_image)
                
                # Load the image using OpenCV
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Convert the image to RGB (OpenCV loads images in BGR by default)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Display the image in a subplot
                    plt.subplot(5, 5, count)  # Adjust grid size (e.g., 5x5 for up to 25 images)
                    plt.imshow(image)
                    plt.title(f"{category[:10]} - {subdir[:10]} ({idx + 1})", fontsize=10)  # Shortened title to avoid overlap
                    plt.axis('off')
                    count += 1  # Increment subplot index
            
            # Stop if the subplot grid is filled
            if count > 25:
                break
    if count > 25:
        break

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Add space between subplots

plt.tight_layout()
plt.show()
