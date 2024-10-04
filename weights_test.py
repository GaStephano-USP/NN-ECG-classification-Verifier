from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import onnxruntime as ort
import tensorflow as tf
import numpy as np
from base_model import *
from dataset_utils.read_MIT_dataset import *
from ensamble.fusion_methods import *
from ensamble.umce import *
from experiment_utils.metrics import *
from oversampling.reduce_imbalance import *
from onnx import numpy_helper
import onnx

if __name__ == '__main__':
   
    keras_model = tf.keras.models.load_model('keras_model.keras')
    onnx_model_path = 'onnx_model.onnx'
    onnx_model = onnx.load(onnx_model_path)
    n = 0
    for layer in keras_model.layers: 
        print(layer.get_config(), layer.get_weights())
        n = n + 1
        if n == 2:
            break
    INTIALIZERS = onnx_model.graph.initializer
    Weight=[]
    n = 0
    print ("COMEÇOU ONNX")
    for initializer in INTIALIZERS:
        W= numpy_helper.to_array(initializer)
        Weight.append(W)
        n = n + 1
        if n == 1:
            break
        
    print(Weight)
    print(layer.get_weights())