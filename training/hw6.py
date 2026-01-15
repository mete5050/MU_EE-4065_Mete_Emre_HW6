import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras import layers, models

# --- AYARLAR ---
BATCH_SIZE = 64
EPOCHS = 5  # Sonuçları iyileştirmek için 10 yapabilirsin
NUM_CLASSES = 10
IMG_SIZE = 32  # En küçük desteklenen boyut

# Çıktı klasörü
OUTPUT_DIR = "stm32_final_models"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"TensorFlow Version: {tf.__version__}")
print("Mode: ULTRA-COMPACT (Alpha=0.35 + Int8 Quantization)")

# --- 1. VERİ HAZIRLAMA ---
def prepare_data():
    print("Veri seti hazırlanıyor...")
    (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)

    def preprocess_image(image, label):
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.image.grayscale_to_rgb(image) # Transfer learning için şart
        image = image / 255.0
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_ds = train_ds.map(preprocess_image).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

# --- 2. C++ DOSYA OLUŞTURUCU ---
def convert_tflite_to_cc(tflite_path, cc_path, model_name):
    with open(tflite_path, 'rb') as f:
        data = f.read()
    
    var_name = f"g_model_{model_name}"
    
    with open(cc_path, 'w') as f:
        f.write(f'#include <cstdint>\n\n')
        f.write(f'// Model Size: {len(data)} bytes\n')
        f.write(f'const unsigned int {var_name}_len = {len(data)};\n')
        f.write(f'alignas(16) const unsigned char {var_name}[] = {{\n')
        
        for i, byte in enumerate(data):
            f.write(f'0x{byte:02x}, ')
            if (i + 1) % 12 == 0:
                f.write('\n')
        
        f.write('\n};\n')
    print(f"C++ Array oluşturuldu: {cc_path} (Boyut: {len(data)} bytes)")

# --- 3. ANA İŞLEM DÖNGÜSÜ ---
def train_and_shrink_all():
    train_ds, val_ds = prepare_data()
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    # Quantization için örnek veri üreteci (Representative Dataset)
    def representative_data_gen():
        for input_value, _ in train_ds.take(100):
            yield [input_value]

    # Modelleri Tanımla
    # EfficientNet ve ResNet mimari gereği çok küçülemez (parametreleri sabittir).
    # Ancak MobileNetV2 'alpha' parametresi ile "Mini" versiyona dönüşebilir.
    models_config = [
        {"name": "MobileNetV2_0.35", "class": MobileNetV2, "alpha": 0.35}, # EN ÖNEMLİSİ BU
        {"name": "EfficientNetB0",   "class": EfficientNetB0, "alpha": 1.0},
        {"name": "ResNet50",         "class": ResNet50,       "alpha": 1.0}
    ]

    results = {}

    for config in models_config:
        name = config["name"]
        print(f"\n================ {name} İŞLENİYOR ================")
        
        # 1. Modeli Kur (Alpha parametresine dikkat)
        kwargs = {'weights': 'imagenet', 'include_top': False, 'input_shape': input_shape}
        if name.startswith("MobileNet"):
            kwargs['alpha'] = config["alpha"] # Model küçültme burada yapılıyor
            
        base_model = config["class"](**kwargs)
        base_model.trainable = False 

        # En basit kafa yapısı (Header) - Parametre tasarrufu için
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])

        # 2. Modeli Eğit
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"{name} eğitiliyor...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
        results[name] = history.history['val_accuracy'][-1]

        # 3. Keras Olarak Kaydet (Yedek)
        model.save(os.path.join(OUTPUT_DIR, f"{name}.h5"))

        # 4. TFLite Dönüştürme ve Sıkıştırma (Quantization)
        print(f"{name} sıkıştırılıyor (Quantization)...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # En iyi sıkıştırma ayarları
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        
        # İsteğe bağlı: Sadece tamsayı işlemciler için (Hata verirse bu satırı kapat)
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()

        # Kaydet
        tflite_filename = f"{name}_quantized.tflite"
        tflite_path = os.path.join(OUTPUT_DIR, tflite_filename)
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"--> {tflite_filename} boyutu: {size_mb:.2f} MB")

        # 5. C++ Header Oluştur
        cc_filename = f"{name}_data.cc"
        convert_tflite_to_cc(tflite_path, os.path.join(OUTPUT_DIR, cc_filename), name.lower().replace(".", "_"))

    print("\n--- SONUÇLAR VE BOYUTLAR ---")
    for name, acc in results.items():
        print(f"{name} Accuracy: {acc:.4f}")
    print(f"Dosyalar '{OUTPUT_DIR}' klasöründe.")

if __name__ == "__main__":
    train_and_shrink_all()
