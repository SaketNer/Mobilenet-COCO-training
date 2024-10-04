import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# Constants
IMG_HEIGHT, IMG_WIDTH = 240, 240
TFLITE_MODEL_PATH = "./modelV3.tflite"  # Path to the quantized TFLite model
TEST_DIR = "./temp"

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

    # Scale the image to [0, 255] and convert to uint8
    img_array = np.clip(img_array, 0, 255)  # Ensure values are in the correct range
    img_array = img_array.astype(np.uint8)  # Convert to uint8
    

    # Set the tensor to the input
    interpreter.set_tensor(input_details[0]["index"], img_array)

    # Invoke the interpreter
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


# Get class labels from the training generator
class_indices = {
    "bottle": 0,
    "car": 1,
    "cat": 2,
    "chair": 3,
    "dog": 4,
    "laptop": 5,
    "person": 6,
}
class_labels = list(class_indices.keys())

# Test images
test_images = os.listdir(TEST_DIR)

# Iterate over test images and make predictions
for img_name in test_images:
    img_path = os.path.join(TEST_DIR, img_name)
    predictions = load_and_predict(img_path)

    # Normalize the output if needed (softmax expected)
    predictions = predictions[0]  # Get the predictions for the batch

    # If the model uses softmax, it should already be normalized
    # Ensure the values are probabilities
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    predicted_confidence = predictions[
        predicted_index
    ]  # Get confidence for the predicted class

    # Display the image and the prediction
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}, confidence: {predicted_confidence:.2f}")
    plt.axis("off")
    plt.show()
