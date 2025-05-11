from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Load the saved model
model = load_model("euroSAT_cnn_model.h5")
print("Model loaded successfully!")

# Example: Predict on a new image
img_path = "D:\\college\\PROJECT\\download.jpeg"
  # Replace with an actual image path
img = load_img(img_path, target_size=(64, 64))  # Resize to match input size
img_array = img_to_array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

prediction = model.predict(img_array)
predicted_label = np.argmax(prediction)  # Get the class index

label_to_category = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake"
}
predicted_category = label_to_category[predicted_label]

# Print the result
print(f"Predicted category: {predicted_category}")
