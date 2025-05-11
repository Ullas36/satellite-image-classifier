import json
import random

# File paths
image_labels_path = r"D:\college\PROJECT\EuroSAT_image_labels.json"
train_split_path = r"D:\college\PROJECT\EuroSAT_train.json"
val_split_path = r"D:\college\PROJECT\EuroSAT_val.json"
test_split_path = r"D:\college\PROJECT\EuroSAT_test.json"

# Load the image-label mappings
with open(image_labels_path, 'r') as json_file:
    image_label_pairs = json.load(json_file)

# Shuffle the data to ensure randomness
random.shuffle(image_label_pairs)

# Define split ratios
train_ratio = 0.7  # 70% training
val_ratio = 0.15   # 15% validation
test_ratio = 0.15  # 15% testing

# Calculate split sizes
total_count = len(image_label_pairs)
train_count = int(total_count * train_ratio)
val_count = int(total_count * val_ratio)

# Split the data
train_set = image_label_pairs[:train_count]
val_set = image_label_pairs[train_count:train_count + val_count]
test_set = image_label_pairs[train_count + val_count:]

# Save the splits to JSON files
with open(train_split_path, 'w') as json_file:
    json.dump(train_set, json_file, indent=4)

with open(val_split_path, 'w') as json_file:
    json.dump(val_set, json_file, indent=4)

with open(test_split_path, 'w') as json_file:
    json.dump(test_set, json_file, indent=4)

# Print summary
print(f"Total images: {total_count}")
print(f"Training set: {len(train_set)} images")
print(f"Validation set: {len(val_set)} images")
print(f"Test set: {len(test_set)} images")
print("Dataset splits saved successfully!")
