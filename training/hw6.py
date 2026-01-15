import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras import layers, models

# --- AYARLAR ---
BATCH_SIZE = 64
EPOCHS = 5  # Ödev için bu sayıyı artırabilirsin
NUM_CLASSES = 10
IMG_SIZE = 32  # Transfer learning modelleri için min 32x32 gerekli

# Çıktı klasörü oluştur
OUTPUT_DIR = "stm32_models_quantized" # Klasör adını değiştirdim karışmasın diye
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. VERİ HAZIRLAMA (Listing 13.16 Referanslı) ---
def prepare_data():
    print("Veri seti yükleniyor ve işleniyor...")
    (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

    # One-hot encoding
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)

    # Veri işleme fonksiyonu (Resizing + Grayscale to RGB)
    def preprocess_image(image, label):
        # Boyut ekle: (28, 28) -> (28, 28, 1)
        image = tf.expand_dims(image, axis=-1)
        # Boyutlandır: (32, 32, 1)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        # RGB'ye çevir: (32, 32, 3) - Transfer learning için şart
        image = tf.image.grayscale_to_rgb(image)
        # Normalize et: [0, 1]
        image = image / 255.0
        return image, label

    # tf.data pipeline oluşturma
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_ds = train_ds.map(preprocess_image).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

# --- 2. C++ DÖNÜŞTÜRME FONKSİYONU ---
def convert_tflite_to_cc(tflite_path, cc_path, model_name):
    with open(tflite_path, 'rb') as f:
        data = f.read()
    
    var_name = f"g_model_{model_name}"
    
    with open(cc_path, 'w') as f:
        f.write(f'#include <cstdint>\n\n')
        f.write(f'// Model boyutu: {len(data)} bytes\n')
        f.write(f'const unsigned int {var_name}_len = {len(data)};\n')
        f.write(f'alignas(16) const unsigned char {var_name}[] = {{\n')
        
        for i, byte in enumerate(data):
            f.write(f'0x{byte:02x}, ')
            if (i + 1) % 12 == 0:
                f.write('\n')
        
        f.write('\n};\n')
    print(f"C++ dosyası oluşturuldu: {cc_path}")

# --- 3. MODEL EĞİTİM VE DÖNÜŞTÜRME DÖNGÜSÜ ---
def train_and_convert_all():
    train_ds, val_ds = prepare_data()
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    # --- QUANTIZATION İÇİN DATA JENERATÖRÜ ---
    # Modelin değer aralıklarını öğrenmesi için eğitim setinden örnekler alıyoruz
    def representative_data_gen():
        for input_value, _ in train_ds.take(100): # 100 batch örnekle
            # Model giriş olarak float32 bekler
            yield [input_value]

    models_to_train = {
        "ResNet50": ResNet50,
        "EfficientNetB0": EfficientNetB0,
        "MobileNetV2": MobileNetV2 
    }

    results = {}

    for name, model_func in models_to_train.items():
        print(f"\n--- {name} Modeli Başlatılıyor ---")
        
        base_model = model_func(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False 

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print(f"{name} eğitiliyor...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
        results[name] = history.history['val_accuracy'][-1]

        # 1. Keras Modelini Kaydet
        h5_path = os.path.join(OUTPUT_DIR, f"{name}_mnist.h5")
        model.save(h5_path)
        print(f"{name} H5 kaydedildi.")

        # 2. TFLite Dönüşümü (QUANTIZATION EKLENDİ)
        print(f"{name} Quantization ile dönüştürülüyor...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # --- KRİTİK DEĞİŞİKLİK BURADA ---
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        # Opsiyonel: Tam sayı zorlaması (gerekirse açılabilir ama bazen hata verir)
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        tflite_model = converter.convert()

        tflite_path = os.path.join(OUTPUT_DIR, f"{name}_mnist_quantized.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"{name} TFLite (Quantized) oluşturuldu. Boyut: {len(tflite_model) / 1024:.2f} KB")

        # 3. C++ Header Dönüşümü
        cc_path = os.path.join(OUTPUT_DIR, f"{name}_model_data.cc")
        convert_tflite_to_cc(tflite_path, cc_path, name.lower())

    print("\n--- TÜM İŞLEMLER TAMAMLANDI ---")
    print("Sonuçlar (Validation Accuracy):")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
    print(f"\nTüm dosyalar '{OUTPUT_DIR}' klasöründe hazır.")

# --- ÇALIŞTIR ---
if __name__ == "__main__":
    train_and_convert_all()