import os
import json

# Path to the dataset
dataset_path = r"D:\college\PROJECT\EuroSAT\2750"

# List subfolders (categories)
categories = sorted(os.listdir(dataset_path))  # Sort for consistent ordering

# Map each subfolder (category) to a unique label
category_labels = {category: idx for idx, category in enumerate(categories)}

# Prepare image-label pairs
image_label_pairs = []

for category in categories:
    category_path = os.path.join(dataset_path, category)
    
    # List all images in the subfolder
    image_files = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Assign the label for each image
    for image_file in image_files:
        image_path = os.path.join(category_path, image_file)
        label = category_labels[category]
        image_label_pairs.append({"image_path": image_path, "label": label})

# Save the mappings to a JSON file
output_label_path = r"D:\college\PROJECT\EuroSAT_image_labels.json"
with open(output_label_path, 'w') as json_file:
    json.dump(image_label_pairs, json_file, indent=4)

# Save the category-to-label mapping
output_category_path = r"D:\college\PROJECT\EuroSAT_category_labels.json"
with open(output_category_path, 'w') as json_file:
    json.dump(category_labels, json_file, indent=4)

print(f"Image-label mappings saved to {output_label_path}")
print(f"Category-label mappings saved to {output_category_path}")
