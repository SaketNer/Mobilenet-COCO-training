import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# Constants
IMG_HEIGHT, IMG_WIDTH = 240, 240
TFLITE_MODEL_PATH = './mobilenet_cats_laptops_quantized.tflite'  # Path to the quantized TFLite model
TEST_DIR = './Dataset/Laptop'  # Update with your test images directory

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess and predict on a single image
def load_and_predict(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale to [0, 1]

    # Set the tensor to the input
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

    # Invoke the interpreter
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Get class labels from the training generator
class_indices = {
    "cat": 0,
    "laptop": 1
}
class_labels = list(class_indices.keys())

# Test images
test_images = os.listdir(TEST_DIR)

# Iterate over test images and make predictions
for img_name in test_images:
    img_path = os.path.join(TEST_DIR, img_name)
    predictions = load_and_predict(img_path)

    # Get the predicted class index and label
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_index]

    # Display the image and the prediction
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_label}')
    plt.axis('off')
    plt.show()
