# Satellite Image Classifier

A complete deep learning pipeline for land cover classification using the **EuroSAT** dataset (based on Sentinel-2 satellite imagery). This project includes scripts for downloading the dataset, preprocessing and normalizing images, splitting the data into subsets, performing runtime data augmentation, training a Convolutional Neural Network (CNN) model using TensorFlow/Keras, and running inference on new satellite images.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Details](#dataset-details)
3. [Repository Structure](#repository-structure)
4. [Prerequisites & Installation](#prerequisites--installation)
5. [Pipeline Workflow & Usage](#pipeline-workflow-and-usage)
   - [Step 1: Verify Installation](#step-1-verify-installation)
   - [Step 2: Download the Dataset](#step-2-download-the-dataset)
   - [Step 3: Visualize the Dataset](#step-3-visualize-the-dataset)
   - [Step 4: Label Generation](#step-4-label-generation)
   - [Step 5: Image Normalization](#step-5-image-normalization)
   - [Step 6: Train-Val-Test Split](#step-6-train-val-test-split)
   - [Step 7: Data Augmentation (Optional/Verification)](#step-7-data-augmentation-optionalverification)
   - [Step 8: Model Training](#step-8-model-training)
   - [Step 9: Model Evaluation & Prediction](#step-9-model-evaluation--prediction)
6. [CNN Architecture Details](#cnn-architecture-details)

---

## Project Overview

Satellite image classification is crucial for environmental monitoring, urban planning, agriculture, and forestry tracking. This repository houses a lightweight yet robust pipeline that uses Keras and TensorFlow to classify satellite patch images into one of 10 land cover categories. 

The pipeline structure is modular, guiding you from initial data retrieval to deploying the trained CNN model for custom inference.

---

## Dataset Details

The dataset used in this project is **EuroSAT**, which consists of **27,000** labeled Sentinel-2 satellite patches, each measuring **64x64 pixels** with 3 channels (RGB).

The dataset is categorized into **10 classes**:
| Class ID | Land Cover Category | Description |
|:---:|---|---|
| `0` | **AnnualCrop** | Annual agricultural crops |
| `1` | **Forest** | Forested regions and dense tree canopies |
| `2` | **HerbaceousVegetation** | Grasslands, wild shrublands, and herbaceous cover |
| `3` | **Highway** | Expressways, main roads, and intercity highways |
| `4` | **Industrial** | Factories, commercial warehouses, and industrial infrastructure |
| `5` | **Pasture** | Meadowlands and cattle grazing fields |
| `6` | **PermanentCrop** | Vineyards, orchards, and perennial agricultural fields |
| `7` | **Residential** | Housing areas, suburban neighborhoods, and urban zones |
| `8` | **River** | Rivers, streams, and canals |
| `9` | **SeaLake** | Lakes, seas, and large open bodies of water |

---

## Repository Structure

```
├── EuroSAT/                              # Extracted raw dataset folder
├── EuroSAT_Normalized/                   # Preprocessed and normalized image outputs
├── EuroSAT_category_labels.json         # Mapping of class names to integer labels
├── EuroSAT_image_labels.json            # Mapping of image paths to integer labels
├── EuroSAT_train.json                    # Metadata for training split
├── EuroSAT_val.json                      # Metadata for validation split
├── EuroSAT_test.json                     # Metadata for test split
├── euroSAT_cnn_model.h5                  # Trained CNN model file
│
├── verify_libraries.py                   # Script to check environment dependencies
├── download dataset.py                   # Script to download & extract EuroSAT zip
├── visualize_dataset.py                  # Script to display grid samples of dataset classes
├── generate_labels.py                    # Script to parse files and write labels metadata
├── normalize_images.py                   # Script to scale pixel values to [0, 1]
├── split_dataset.py                      # Script to partition data into train/val/test splits
├── load_dataset.py                       # Python module to load datasets into numpy arrays
├── data_augmentation.py                  # Script implementing ImageDataGenerator for data augmentation
├── CNN Arch.py                           # CNN architecture creation, training, and evaluation
├── load_model_test.py                    # Script to load the trained model and run custom predictions
└── README.md                             # Project documentation (this file)
```

---

## Prerequisites & Installation

To run this project, make sure you have Python 3.8+ installed. You can install the required Python packages using `pip`:

```bash
pip install tensorflow numpy pandas matplotlib opencv-python tqdm requests
```

Alternatively, you can verify your installation by running:
```bash
python verify_libraries.py
```
If all dependencies are properly installed, the console will print:
`All libraries are installed successfully!`

---

## Pipeline Workflow and Usage

### Step 1: Verify Installation
Ensure all libraries are present:
```bash
python verify_libraries.py
```

### Step 2: Download the Dataset
Fetch the dataset directly from the DFKI servers:
```bash
python "download dataset.py"
```
*Note: This downloads `EuroSAT.zip` (RGB version) and extracts it into the `EuroSAT` directory.*

### Step 3: Visualize the Dataset
Display sample images from each folder:
```bash
python visualize_dataset.py
```
This generates a matplotlib grid with sample patch images for visual verification.

### Step 4: Label Generation
Scan the downloaded files to create index JSON mappings:
```bash
python generate_labels.py
```
Generates two files:
1. `EuroSAT_image_labels.json` containing the mapping: `{"image_path": "...", "label": 0-9}`
2. `EuroSAT_category_labels.json` mapping categories to labels: `{"AnnualCrop": 0, "Forest": 1, ...}`

### Step 5: Image Normalization
Scale and process the image dimensions:
```bash
python normalize_images.py
```
Reads each image using OpenCV, scales the pixel values to `[0.0, 1.0]`, scales it back to 8-bit uint8 for storage if desired, and writes the output directory structure inside `EuroSAT_Normalized/`.

### Step 6: Train-Val-Test Split
Create dataset splits for training and evaluation:
```bash
python split_dataset.py
```
Splits the image-label mappings into:
- **70% Training** (`EuroSAT_train.json`)
- **15% Validation** (`EuroSAT_val.json`)
- **15% Testing** (`EuroSAT_test.json`)

### Step 7: Data Augmentation (Optional/Verification)
Verify data generator pipelines and augmentation parameters:
```bash
python data_augmentation.py
```
This script configures a standard Keras `ImageDataGenerator` with random rotation, width/height shifts, shear, zoom, and horizontal flips, reading datasets directly via `pandas` DataFrames.

### Step 8: Model Training
Train the CNN model:
```bash
python "CNN Arch.py"
```
This script reads the JSON splits using `load_dataset.py`, loads and scales the images, builds the CNN architecture, compiles it using the Adam optimizer with a learning rate of `0.001`, trains it over `10` epochs, prints validation/test metrics, and saves the trained model parameters to `euroSAT_cnn_model.h5`.

### Step 9: Model Evaluation & Prediction
Perform inference on a single sample image:
```bash
python load_model_test.py
```
You can edit the image path in `load_model_test.py` to target any new satellite patch. The script will resize the input image to **64x64**, normalize the pixels, feed it to the trained CNN model, and output the predicted category.

---

## CNN Architecture Details

The model is built using Keras's sequential API:

1. **Input Layer**: Accepts `(64, 64, 3)` RGB images.
2. **Convolutional Layer 1**: `32` filters of size `3x3` with `ReLU` activation.
3. **Max Pooling Layer 1**: Pool size of `2x2`.
4. **Convolutional Layer 2**: `64` filters of size `3x3` with `ReLU` activation.
5. **Max Pooling Layer 2**: Pool size of `2x2`.
6. **Flatten Layer**: Flattens the multidimensional feature maps into a 1D vector.
7. **Dense (Fully Connected) Layer 1**: `128` units with `ReLU` activation.
8. **Dropout Layer**: Rate of `0.5` to prevent overfitting during training.
9. **Dense (Output) Layer**: `10` units with `Softmax` activation mapping to each of the land cover categories.

### Optimizer and Loss
- **Optimizer**: Adam (learning rate = `0.001`)
- **Loss Function**: Categorical Crossentropy
- **Evaluation Metric**: Accuracy
