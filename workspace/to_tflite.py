import tensorflow as tf

WINDOW = 100

model = tf.keras.models.load_model(
    f"models/csi_hr_best_{WINDOW}.keras",
    compile=False
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(f"models/csi_hr_best_{WINDOW}.tflite", "wb") as f:
    f.write(tflite_model)
