import numpy as np
import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('./model.keras')

# Set up the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the optimization flag for quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Specify input and output types to be int8
converter.target_spec.supported_types = [tf.int8]

# Define the representative dataset generator
def representative_dataset_gen():
    for _ in range(100):
        # Replace these with valid dimensions for your model
        input_data = np.random.rand(1, 240, 240, 3).astype(np.float32)  # Adjust the shape if needed
        yield [input_data]

converter.representative_dataset = representative_dataset_gen
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted and saved as model_quantized.tflite")

# Verify the model
interpreter = tf.lite.Interpreter(model_path='model_quantized.tflite')
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Check types
print("Input type:", input_details[0]['dtype'])
print("Output type:", output_details[0]['dtype'])
