import tensorflow as tf

# carica il tuo modello Keras
model = tf.keras.models.load_model("models/csi_hr_best_200.keras")

# crea un convertitore TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# abilita ottimizzazione (quantizzazione automatica)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# converti in TFLite
tflite_model = converter.convert()

# salva il modello convertito
with open("models/csi_hr_best_200.tflite", "wb") as f:
    f.write(tflite_model)
