#!/usr/bin/env python
# test_occlusion.py - Test the keras_interpret library with real models and images

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from urllib.request import urlretrieve

# Add the library to the path if it's not installed
sys.path.insert(0, os.path.abspath('.'))

# Set keras backend to numpy
os.environ["KERAS_BACKEND"] = "numpy"

# Import keras
import keras

# Import library components
from interpretabilityLibrary.methods.occlusionSensitivity import OcclusionSensitivity
from interpretabilityLibrary.plot.vizualize import vizualize

def load_test_image(filepath="test_image.jpg", target_size=(224, 224)):
    """Load an existing image or download one if not found."""
    if not os.path.exists(filepath):
        print(f"Test image {filepath} not found, downloading a sample image...")
        # Use a reliable image URL from TensorFlow's servers
        img_url = "https://storage.googleapis.com/tensorflow/keras-applications/tests/elephant.jpg"
        try:
            urlretrieve(img_url, filepath)
            print(f"Downloaded sample image to {filepath}")
        except Exception as e:
            print(f"Failed to download image: {e}")
            # Create a simple gradient image as fallback
            print("Creating a fallback test image")
            img_array = np.zeros(target_size + (3,), dtype=np.uint8)
            for i in range(target_size[0]):
                for j in range(target_size[1]):
                    img_array[i, j, 0] = i % 256  # R
                    img_array[i, j, 1] = j % 256  # G
                    img_array[i, j, 2] = (i + j) % 256  # B
            img = Image.fromarray(img_array)
            img.save(filepath)
            return img_array
    
    # Load and preprocess the image
    img = Image.open(filepath).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    
    return img_array

def load_test_image(filepath="test_image.jpg", target_size=(224, 224)):
    """Load an existing image or download one if not found."""
    if not os.path.exists(filepath):
        print(f"Test image {filepath} not found, downloading a sample image...")
        # Use a reliable image URL - specifically an elephant image
        img_url = "https://storage.googleapis.com/tensorflow/keras-applications/tests/elephant.jpg"

        urlretrieve(img_url, filepath)
        print(f"Downloaded sample image to {filepath}")

        # Load and preprocess the image
        img = Image.open(filepath).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)

        return img_array


def preprocess_for_resnet50(img_array):
    """Apply the correct preprocessing for ResNet50."""
    # Expand dimensions to create a batch
    x = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    # Convert RGB to BGR (ResNet50 was trained with BGR images)
    x = x[..., ::-1]
    
    # Zero-center by mean pixel values (specific to ResNet50)
    x[..., 0] -= 103.939  # B channel
    x[..., 1] -= 116.779  # G channel
    x[..., 2] -= 123.68   # R channel
    
    return x

def load_or_create_model(filepath="test_model.keras"):
    """Load a real vision model."""
    if os.path.exists(filepath):
        print(f"Loading model from {filepath}")
        model = keras.models.load_model(filepath)
        return model
    
    print(f"Model {filepath} not found, loading a pre-trained ResNet50...")
    
    # Load ResNet50 with pre-trained weights
    model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=True,
    weights='imagenet'
)
    
    # No need to save the model to disk
    return model

def test_occlusion_sensitivity():
    """Test OcclusionSensitivity implementation with real models and images."""
    print("Testing OcclusionSensitivity with real data...")
    
    # Load a test image
    print("Loading test image...")
    img_array = load_test_image()
    img_array_for_display = img_array.copy()
    
    # Preprocess image for model
    
    # Load or create model
    print("Loading/creating model...")
    model = load_or_create_model()
    
    # Get prediction
    print("Getting model prediction...")
    preds = model.predict(x)  # Using predict() to be compatible with different model types
    top_pred_idx = np.argmax(preds[0])
    print(f"Predicted class index: {top_pred_idx}")
    print(f"Prediction confidence: {preds[0][top_pred_idx]:.4f}")
    
    
    # Create explainer with large window
    print("\nGenerating occlusion map with 32x32 window...")
    start_time = time.time()
    explainer_large = OcclusionSensitivity(
        model, 
        window_size=(32, 32), 
        stride=(16, 16),
        occlusion_value=0.0
    )
    
    # Generate explanation
    explanation_large = explainer_large.explain(x, targets=top_pred_idx)
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f} seconds")
    
    # Validate explanation
    assert explanation_large.attributions.shape == (1, 224, 224, 1), f"Unexpected shape: {explanation_large.attributions.shape}"
    
    # Save vizualization
    print("Saving vizualization...")
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Large Window (32x32) Occlusion Sensitivity", fontsize=16)
    vizualize(explanation_large, img_array_for_display / 255.0, save_path="occlusion_large_window.png")
    
    print("\nOcclusion sensitivity tests completed successfully!")
    return True


def main():
    """Run all tests."""
    print("=== Testing keras_interpret library ===")
    print(f"Keras backend: {keras.config.backend()}")
    print(f"Keras version: {keras.__version__}")
    
    # Test occlusion sensitivity
    test_occlusion_sensitivity()
    
    # Test different target classes
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()