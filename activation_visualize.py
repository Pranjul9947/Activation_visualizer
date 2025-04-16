import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import cv2
import os

# --- Load Model ---
def load_model(model_json, model_weights):
    with open(model_json, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights)
    print("[INFO] Model loaded successfully.")
    return model

# --- Preprocess Image ---
def preprocess_input(img_path, target_size=(48, 48)):
    img = image.load_img(img_path, target_size=target_size, color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# --- Visualize Activation Maps ---
def visualize_activations(model, img_array):
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img_array)

    for i, activation in enumerate(activations):
        num_filters = activation.shape[-1]
        size = activation.shape[1]
        cols = 8
        rows = num_filters // cols
        print(f"[INFO] Layer {i+1}: {model.layers[i].name}, filters: {num_filters}")
        
        display_grid = np.zeros((size * rows, size * cols))
        for row in range(rows):
            for col in range(cols):
                filter_index = row * cols + col
                if filter_index < num_filters:
                    act_img = activation[0, :, :, filter_index]
                    act_img -= act_img.mean()
                    act_img /= act_img.std() + 1e-5
                    act_img *= 64
                    act_img += 128
                    act_img = np.clip(act_img, 0, 255).astype('uint8')
                    display_grid[row * size : (row + 1) * size,
                                 col * size : (col + 1) * size] = act_img

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(f"Layer {i+1} - {model.layers[i].name}")
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

# --- Main ---
if __name__ == "__main__":
    model_path = "model_a1.json"
    weights_path = "model.weights.h5"
    img_path = "test.jpg"

    model = load_model(model_path, weights_path)
    img_array = preprocess_input(img_path)
    visualize_activations(model, img_array)
