import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# Constants
IMG_HEIGHT, IMG_WIDTH = 240, 240
BATCH_SIZE = 64
EPOCHS = 20
NUM_CLASSES = 7
TRAIN_DIR = 'Dataset/Train'

def rgb_to_rgb565(image):
    r = (image[:, :, 0] >> 3) & 0x1F
    g = (image[:, :, 1] >> 2) & 0x3F
    b = (image[:, :, 2] >> 3) & 0x1F
    rgb565 = (r << 11) | (g << 5) | b
    return rgb565.astype(np.uint16)

# Data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',  # Ensures the model uses RGB888
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',  # Ensures the model uses RGB888
    subset='validation'
)

for images, labels in train_generator:
    images_rgb565 = rgb_to_rgb565(images)

for images, labels in validation_generator:
    images_rgb565 = rgb_to_rgb565(images)

# Build the MobileNet model
base_model = MobileNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Reshape((1, 1, -1))(x)  # Reshape to (batch_size, 1, 1, features)
x = Conv2D(NUM_CLASSES, kernel_size=(1, 1), activation='softmax')(x)  # Use Conv2D for classification
x = Reshape((NUM_CLASSES,))(x)  # Flatten back to (batch_size, NUM_CLASSES)

model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
checkpoint = ModelCheckpoint('./mobilenet_cats_laptops.keras', monitor='val_accuracy', save_best_only=True)
model.fit(train_generator, validation_data=validation_generator, epochs=EPOCHS, callbacks=[checkpoint])

# Save the model before quantization
model.save('./model.keras')

# Quantization to INT8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_disable_per_channel_quantization_for_dense_layers = True 

# Provide representative dataset for quantization
def representative_data_gen():
    for i in range(10):
        x, _ = next(train_generator) 
        yield [x]
        print(i)

converter.representative_dataset = representative_data_gen
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# Save the quantized model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Training complete and quantized model saved.")
