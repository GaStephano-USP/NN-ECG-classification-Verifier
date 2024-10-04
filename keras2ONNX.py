import tf2onnx
import keras
import tensorflow as tf
import onnx
import onnxruntime


loaded_model = keras.saving.load_model("keras_model.keras")
output_path = "onnx_model.onnx"

spec = (tf.TensorSpec((None, 187, 1), dtype=tf.dtypes.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(loaded_model, input_signature=spec, output_path=output_path)


onnx_model = onnx.load(output_path)
