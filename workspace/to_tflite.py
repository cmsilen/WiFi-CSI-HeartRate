import tensorflow as tf

model = tf.keras.models.load_model("models/csi_hr_best_200.keras", safe_mode=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Usa ops selezionati TF per layer non supportati da default
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLite standard ops
    tf.lite.OpsSet.SELECT_TF_OPS     # layer complessi come LSTM
]

# Disabilita la lowering dei tensor list ops
converter._experimental_lower_tensor_list_ops = False

# Optional: ottimizzazione per ridurre peso e RAM
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("models/csi_hr_best_200.tflite", "wb") as f:
    f.write(tflite_model)

print("Modello convertito in TFLite con Select TF Ops!")
