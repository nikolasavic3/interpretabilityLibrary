import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from urllib.request import urlretrieve
import keras
from keras.applications.imagenet_utils import decode_predictions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["KERAS_BACKEND"] = "openvino"

from interpretabilityLibrary.methods.occlusionSensitivity import OcclusionSensitivity
from interpretabilityLibrary.plot.vizualize import vizualize

def load_test_image(filepath="test_image.jpg", target_size=(96, 96)):
    img_url = "https://storage.googleapis.com/tensorflow/keras-applications/tests/elephant.jpg"
    urlretrieve(img_url, filepath)
    print(f"Downloaded sample image to {filepath}")
    img = Image.open(filepath).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    return img_array

def test_occlusion_sensitivity():
    print("Testing...")
    
    print("Loading test image...")
    img_size = (96, 96)
    img_array = load_test_image(target_size=img_size)
    img_array_for_display = img_array.copy()
    
    # Display the input image
    plt.figure(figsize=(5, 5))
    plt.imshow(img_array)
    plt.title("Input Image")
    plt.axis('off')
    plt.show()
    
    # Convert to float32 and normalize to [-1, 1] range
    x = img_array.astype(np.float32)
    x = x / 127.5 - 1.0
    x = np.expand_dims(x, axis=0)  # Add batch dimension

    print("Loading/creating model...")
    model = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=True,
        weights='imagenet'
    )

    print("Getting model prediction...")
    preds = model.predict(x)
    top_indices = np.argsort(preds[0])[-3:][::-1]
    top_confidences = preds[0][top_indices]
    class_names = decode_predictions(preds, top=3)[0]
    
    target_class = top_indices[0]  # Use the top predicted class

    print("\nTop 3 predictions:")
    for i, (_, name, conf) in enumerate(class_names):
        print(f"{i+1}. {name} ({top_confidences[i]:.4f})")
    
    # Create explainer
    print("Creating explainer...")
    explainer = OcclusionSensitivity(
        model, 
        window_size=(8, 8),  
        stride=(8, 8),       
        occlusion_value=0.0
    )
    
    # Generate explanation
    print("Generating explanation...")
    start_time = time.time()
    explanation = explainer.explain(x, targets=target_class)
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f} seconds")
    
    # Visualize results
    print("Visualizing results...")
    vizualize(explanation, img_array, save_path="occlusion.png")
    
    print("\nOcclusion sensitivity tests completed successfully!")
    return True

def main():
    print("=== Testing keras_interpret library ===")
    print(f"Keras backend: {keras.config.backend()}")
    
    test_occlusion_sensitivity()
        
if __name__ == "__main__":
    main()
