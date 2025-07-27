import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Set ukuran gambar input untuk model
IMAGE_SIZE = (224, 224)

# Path ke folder dataset
DATASET_PATH = "CROPPED_FOLDER"  # Harus berisi subfolder per kelas

# Inisialisasi ImageDataGenerator untuk augmentasi training dan preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalisasi piksel ke range [0,1]
    rotation_range=20,          # Rotasi acak
    zoom_range=0.2,             # Zoom acak
    width_shift_range=0.2,      # Geser horizontal
    height_shift_range=0.2,     # Geser vertikal
    shear_range=0.2,            # Distorsi shearing
    horizontal_flip=True,       # Balik gambar secara horizontal
    validation_split=0.2        # 20% data digunakan sebagai validasi
)

# Generator data training (80% data)
train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,     # Semua gambar diresize ke IMAGE_SIZE
    batch_size=32,
    class_mode='categorical',   # Karena klasifikasi multi-kelas
    subset='training',          # Ambil subset training
    shuffle=True
)

# Generator data validasi (20% data)
val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Simpan label mapping ke txt
with open("label_map.txt", "w") as f:
    for label, idx in train_gen.class_indices.items():
        f.write(f"{idx}: {label}\n")

# Load model MobileNetV2 tanpa top layer (tanpa head klasifikasi)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze backbone agar tidak di-train ulang

# Tambahkan head klasifikasi baru di atas MobileNetV2
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),                   # Global average pooling untuk hasil akhir dari feature map
    Dense(128, activation='relu'),              # Layer dense dengan 128 neuron
    Dense(train_gen.num_classes, activation='softmax')  # Output layer, jumlah neuron = jumlah kelas
])

# Compile model dengan optimizer, loss, dan metric
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Tambahkan callback untuk menghentikan training saat overfitting dan menyimpan model terbaik
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('mobilenet_gesture_model.h5', save_best_only=True)
]

# Training model dengan data generator
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks
)

# Evaluasi model pada data validasi
loss, acc = model.evaluate(val_gen)
print(f"🎯 Akurasi Validasi: {acc:.2%}")

# Plot grafik training & validation loss dan akurasi
plt.figure(figsize=(12, 5))

# Grafik akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Akurasi')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

# Grafik loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Visualisasi contoh data training
import random
sample_class = random.choice(os.listdir(DATASET_PATH))  # Ambil nama folder acak (kelas)
sample_path = os.path.join(DATASET_PATH, sample_class)
sample_image = random.choice(os.listdir(sample_path))   # Ambil gambar acak dari folder kelas
img_path = os.path.join(sample_path, sample_image)

img = cv2.imread(img_path)
img = cv2.resize(img, IMAGE_SIZE)

plt.figure(figsize=(4, 4))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Contoh data: {sample_class}")
plt.axis("off")
plt.show()
