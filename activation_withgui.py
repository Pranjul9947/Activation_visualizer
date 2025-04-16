import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.preprocessing import image

# --- Load Model ---
def load_model(model_json, model_weights):
    with open(model_json, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights)
    return model

# --- Preprocess Image ---
def preprocess_input(img_path, target_size=(48, 48)):
    img = image.load_img(img_path, target_size=target_size, color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# --- Visualize Activations ---
def visualize_activations(model, img_array):
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name]
    activation_model = Model(inputs=model.input, outputs=[layer.output for layer in conv_layers])
    activations = activation_model.predict(img_array)

    for i, activation in enumerate(activations):
        layer_name = conv_layers[i].name
        num_filters = activation.shape[-1]
        size = activation.shape[1]
        cols = 8
        rows = (num_filters + cols - 1) // cols
        display_grid = np.zeros((size * rows, size * cols))

        for row in range(rows):
            for col in range(cols):
                filter_index = row * cols + col
                if filter_index < num_filters:
                    act_img = activation[0, :, :, filter_index]
                    act_img -= act_img.mean()
                    act_img /= (act_img.std() + 1e-5)
                    act_img *= 64
                    act_img += 128
                    act_img = np.clip(act_img, 0, 255).astype('uint8')
                    display_grid[row * size:(row + 1) * size,
                                 col * size:(col + 1) * size] = act_img

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(f"Layer {i+1}: {layer_name}")
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

# --- GUI Logic ---
class ActivationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Activation Map Visualizer")

        self.model_json = None
        self.weights_file = None
        self.image_file = None
        self.model = None

        # Buttons
        tk.Button(root, text="Load Model JSON", command=self.load_json).pack(pady=5)
        tk.Button(root, text="Load Weights", command=self.load_weights).pack(pady=5)
        tk.Button(root, text="Select Image", command=self.select_image).pack(pady=5)
        tk.Button(root, text="Visualize Activations", command=self.run_visualization).pack(pady=10)

    def load_json(self):
        self.model_json = filedialog.askopenfilename(title="Select model.json", filetypes=[("JSON files", "*.json")])
        if self.model_json:
            messagebox.showinfo("Model JSON", f"Loaded: {self.model_json}")

    def load_weights(self):
        self.weights_file = filedialog.askopenfilename(title="Select model weights", filetypes=[("H5 files", "*.h5")])
        if self.weights_file:
            messagebox.showinfo("Model Weights", f"Loaded: {self.weights_file}")

    def select_image(self):
        self.image_file = filedialog.askopenfilename(title="Select test image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if self.image_file:
            messagebox.showinfo("Image Selected", f"Loaded: {self.image_file}")

    def run_visualization(self):
        if not self.model_json or not self.weights_file or not self.image_file:
            messagebox.showerror("Error", "Please load all required files.")
            return

        try:
            self.model = load_model(self.model_json, self.weights_file)
            img_array = preprocess_input(self.image_file)
            visualize_activations(self.model, img_array)
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))

# --- Run the GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ActivationGUI(root)
    root.mainloop()
