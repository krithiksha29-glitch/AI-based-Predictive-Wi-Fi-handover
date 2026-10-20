import tensorflow as tf
import numpy as np

old_model = tf.keras.models.load_model('handover_predictor.keras')

inputs = tf.keras.Input(shape=(5, 4), batch_size=1)
x = tf.keras.layers.LSTM(32, unroll=False)(inputs)
x = tf.keras.layers.Dense(16, activation='relu')(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
new_model = tf.keras.Model(inputs, outputs)
new_model.set_weights(old_model.get_weights())

converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
# No optimizations, no quantization — plain float32, avoids the WHILE-loop calibration hang entirely
tflite_model = converter.convert()

with open('handover_predictor_int8.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"Saved handover_predictor_int8.tflite")
print(f"Model size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.2f} KB)")