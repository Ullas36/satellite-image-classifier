from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd  # Import pandas
import json

# Paths to the dataset splits and label map
train_split_path = r"D:\college\PROJECT\EuroSAT_train.json"
val_split_path = r"D:\college\PROJECT\EuroSAT_val.json"
label_map_path = r"D:\college\PROJECT\EuroSAT_category_labels.json"

# Load the label map
with open(label_map_path, 'r') as f:
    label_map = json.load(f)

# Reverse the label map to get a numeric-to-string mapping
reverse_label_map = {v: k for k, v in label_map.items()}

# Load training and validation data
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_paths = [item['image_path'] for item in data]
    labels = [item['label'] for item in data]
    return image_paths, labels

train_image_paths, train_labels = load_data(train_split_path)
val_image_paths, val_labels = load_data(val_split_path)

# Convert numeric labels to string labels
train_labels = [reverse_label_map[label] for label in train_labels]
val_labels = [reverse_label_map[label] for label in val_labels]

# Convert to DataFrames
train_df = pd.DataFrame({"filename": train_image_paths, "class": train_labels})
val_df = pd.DataFrame({"filename": val_image_paths, "class": val_labels})

# Create data generators
train_datagen = ImageDataGenerator(
    rotation_range=20,        # Rotate images by 0-20 degrees
    width_shift_range=0.2,    # Horizontal shift
    height_shift_range=0.2,   # Vertical shift
    shear_range=0.2,          # Shear transformation
    zoom_range=0.2,           # Zoom in/out
    horizontal_flip=True,     # Flip images horizontally
    fill_mode='nearest'       # Fill empty pixels
)

val_datagen = ImageDataGenerator()  # No augmentation for validation data

# Image size and batch size
image_size = (64, 64)  # Target size for all images
batch_size = 32        # Number of images per batch

# Create training and validation generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filename",
    y_col="class",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse'  # For string labels
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filename",
    y_col="class",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse'
)

# Verify the generators (optional)
for images, labels in train_generator:
    print(f"Batch of images: {images.shape}, Batch of labels: {labels.shape}")
    break
