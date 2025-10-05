import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Data Paths (project structure के according)
train_dir = 'data/cats_dogs/train'
test_dir = 'data/cats_dogs/test'

# Check अगर folders exist
if not os.path.exists(train_dir):
    print(f"Error: {train_dir} folder नहीं मिला। Kaggle से डाउनलोड करें और extract करें।")
    exit()

# Data Generators with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize for speed
    batch_size=32,
    class_mode='binary'  # Cats=0, Dogs=1
)
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

# CNN Model Build Function (Binary classification)
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')  # Binary के लिए sigmoid
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Model बनाएं
model = build_cnn((150, 150, 3), 1)
model.summary()

# Train Model with Generators
print("Training शुरू हो रही है...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(validation_generator, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Accuracy Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Predictions Plot (कुछ test images)
def plot_predictions(model, generator, class_names, num_images=9):
    x, y = next(generator)  # Batch लें
    predictions = model.predict(x[:num_images])
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = y[:num_images].astype(int)
    
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i+1)
        plt.imshow(x[i])
        color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
        plt.title(f'True: {class_names[int(true_classes[i])]}\nPred: {class_names[int(predicted_classes[i])]}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class_names = ['Cat', 'Dog']
plot_predictions(model, validation_generator, class_names)

print("Cats vs Dogs Project Complete! Model save करने के लिए: model.save('catsdogs_model.h5')")
