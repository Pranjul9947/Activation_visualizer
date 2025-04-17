# 🔍 Activation Map Visualizer

A simple yet powerful tool to **visualize convolutional layer activations** in a CNN (Convolutional Neural Network).  
This helps in understanding **how deep learning models "see" images**, layer by layer.

---

## 📸 Example Use Case

- You have a trained CNN model (e.g., for emotion recognition, object detection, etc.)
- You want to **visualize what each convolutional layer is learning**
- This tool generates beautiful **activation heatmaps** from grayscale input images

---

## 🚀 Features

- Load pre-trained Keras models from `.json` and `.h5` files
- Preprocess input images (grayscale, 48x48)
- Visualize activations of all convolutional layers
- Clean grid layout of filters with color scaling
- Works with TensorFlow backend

---

## 📁 Project Structure
**Activation_visualizer/**
-model_a1.json (Model architecture)
-model.weights.h5 (Model weights)
-test.jpg (Input image to test)
-visualize.py (Python script to visualize the image)
-environment.yml (for creating a conda environment with required dependencies to run on your device)
-README.md (Optional)

## 🛠 Requirements

- Python 3.7+
- TensorFlow (tested on 2.x)
- NumPy
- OpenCV
- Matplotlib
- Keras (via TensorFlow)
- Conda (recommended)

Install all dependencies using:

```bash
conda env create -f environment.yml
conda activate activation-vis
```

## How to Use(activation_withgui.py)

Place your model files in the project directory:
-model_a1.json – architecture
-model.weights.h5 – weights
-Add your test image as test.jpg
-Run the script:

```bash
python visualize.py
```
View the activation maps plotted using matplotlib.

---
