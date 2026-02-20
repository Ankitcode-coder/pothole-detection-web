import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("pothole_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("pothole_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion successful! pothole_model.tflite created.")