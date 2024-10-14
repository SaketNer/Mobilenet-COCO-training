import tensorflow as tf

# Load the TFLite model
tflite_model_path = "./Models/model.tflite"  # Replace with your model path
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

# Allocate tensors
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input layer details
print("Input Layer:")
for input_detail in input_details:
    print(f"Name: {input_detail['name']}")
    print(f"Shape: {input_detail['shape']}")
    print(f"Dtype: {input_detail['dtype']}")
    print(f"Quantization: {input_detail['quantization']}")
    print()

# Print output layer details
print("Output Layer:")
for output_detail in output_details:
    print(f"Name: {output_detail['name']}")
    print(f"Shape: {output_detail['shape']}")
    print(f"Dtype: {output_detail['dtype']}")
    print(f"Quantization: {output_detail['quantization']}")
    print()
