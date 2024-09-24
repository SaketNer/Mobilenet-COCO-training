import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# Constants
IMG_HEIGHT, IMG_WIDTH = 240, 240
BATCH_SIZE = 32
EPOCHS = 3
NUM_CLASSES = 2
TRAIN_DIR = 'Dataset'

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

# Build the MobileNet model
base_model = MobileNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
checkpoint = ModelCheckpoint('./mobilenet_cats_laptops.keras', monitor='val_accuracy', save_best_only=True)
model.fit(train_generator, validation_data=validation_generator, epochs=EPOCHS, callbacks=[checkpoint])

# Save the model before quantization
model.save('./mobilenet_cats_laptops.keras')

# Quantization to INT8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide representative dataset for quantization
def representative_data_gen():
    for i in range(3):
        x, _ = next(train_generator) 
        yield [x]
        print(i)

converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()

# Save the quantized model
with open('mobilenet_cats_laptops_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Training complete and quantized model saved.")
