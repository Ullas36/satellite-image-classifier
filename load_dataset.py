import json
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to the normalized images folder
normalized_folder = "EuroSAT_Normalized"

# Function to load the dataset from JSON
def load_dataset(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    images = []
    labels = []

    for item in data:
        # Load the image using the `image_path` key
        img_path = item['image_path']
        label = item['label']
        
        # Load and preprocess the image
        img = load_img(img_path, target_size=(64, 64))  # Assuming 64x64 images
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(label)
    
    # Normalize the images
    images = np.array(images) / 255.0
    labels = to_categorical(labels, num_classes=10)  # Assuming 10 classes
    
    return images, labels

# Function to prepare the train, validation, and test datasets
def prepare_data():
    x_train, y_train = load_dataset("EuroSAT_train.json")
    x_val, y_val = load_dataset("EuroSAT_val.json")
    x_test, y_test = load_dataset("EuroSAT_test.json")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
