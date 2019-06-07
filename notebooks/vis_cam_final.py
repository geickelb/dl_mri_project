##### Author: Yanrong Ji, Yucheng Zhu, Sharon Zhang, Claudia Xu #####

import os
import cv2
import sys
import glob
import math
import tempfile
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Input,Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # do not use gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] ="3" # Supress tensorflow warning

# # Fix tensorflow GPU allocation
# #%% GPU memory fix
# def get_session(gpu_fraction=0.5):    
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)    
#     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# keras.backend.set_session(get_session())

# # Qucik check if it is using GPU
# from tensorflow.python.client import device_lib
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
     

# print("Tensorflow Version:", tf.__version__)
# print(tf.keras.__version__)
# print(get_available_gpus())



def find_layer_idx(model, layer_name):
    """Looks up the layer index corresponding to `layer_name` from `model`.
    Args:
        model: The `keras.models.Model` instance.
        layer_name: The name of the layer to lookup.
    Returns:
        The layer index if found. Raises an exception otherwise.
    """
    layer_idx = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            layer_idx = idx
            break

    if layer_idx is None:
        raise ValueError("No layer with name '{}' within the model".format(layer_name))
    return layer_idx

def apply_modifications(model, custom_objects=None):
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)

def convert_model(current_model):
    # Utility to search for layer index by name. 
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = find_layer_idx(current_model, 'softmax')

    # Swap softmax with linear
    current_model.layers[layer_idx].activation = activations.linear
    new_model = apply_modifications(current_model)
    return new_model

def visualize_cam(model, img_path, conv_layer = 'conv_7b', size = (300,300), hif = .8, bw = False):
    original_img = cv2.imread(img_path, 3)
    
    img = image.load_img(img_path, target_size=size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = x/255

    # Prediction
    preds = model.predict(x)
    print(preds)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    print("* Predicted class for color: ", argmax)
    
    

    last_conv_layer = model.get_layer(conv_layer)
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    
    for i in range(last_conv_layer.output_shape[3]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * hif + original_img

    if bw:
        # grayscale
        original_img_bw = cv2.imread(img_path, 0)
        img_bw = cv2.resize(original_img_bw,(300,300))
        img_bw = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB);
        
        x_bw = image.img_to_array(img_bw)
        x_bw = np.expand_dims(x_bw, axis=0)
        x_bw = preprocess_input(x_bw)
        x_bw = x_bw/255

        # Prediction
        preds_bw = model.predict(x_bw)
        print(preds_bw)
        argmax_bw = np.argmax(preds_bw[0])
        output_bw = model.output[:, argmax_bw]
        print("* Predicted class for bw: ", argmax_bw)

        last_conv_layer = model.get_layer(conv_layer)
        grads_bw = K.gradients(output_bw, last_conv_layer.output)[0]
        pooled_grads_bw = K.mean(grads_bw, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads_bw, last_conv_layer.output[0]])
        pooled_grads_value_bw, conv_layer_output_value_bw = iterate([x])

        for i in range(last_conv_layer.output_shape[3]):
            conv_layer_output_value_bw[:, :, i] *= pooled_grads_value_bw[i]
        
        heatmap_bw = np.mean(conv_layer_output_value_bw, axis=-1)
        heatmap_bw = np.maximum(heatmap_bw, 0)
        heatmap_bw /= np.max(heatmap_bw)

        heatmap_bw = cv2.resize(heatmap_bw, (original_img_bw.shape[1], original_img_bw.shape[0]))
        heatmap_bw = np.uint8(255 * heatmap_bw)
        heatmap_bw = cv2.applyColorMap(heatmap_bw, cv2.COLORMAP_BONE)
        original_img_bw = cv2.cvtColor(original_img_bw, cv2.COLOR_GRAY2RGB);
        superimposed_img_bw = heatmap_bw * hif + original_img_bw

        return [original_img, superimposed_img, superimposed_img_bw]

    else:
        return [original_img, superimposed_img, 0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type = str, help = "Path of an image to run the network on")
    parser.add_argument("--output_path", type = str, default = "./", help = "Output path + basename")
    parser.add_argument("--model_path", type = str, default = "./model.h5", help = "Path of the trained (and converted) model")
    parser.add_argument("--convert_model", type = bool, default = False, help = 'Convert model softmax layer to linear layer?')
    parser.add_argument("--output_all", type = bool, default = False, help = 'Output all images (original+color+bw)?')
    parser.add_argument("--conv_layer_name", type = str, default = 'conv', help = 'Name of last convolutional layer?')
    parser.add_argument("--image_size", type = int, default = 300, help = 'Image size of model (an int)?')
    parser.add_argument("--hif", type = float, default = .8, help = 'Heatmap factor?')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    output_path = args.output_path
    print('* Loading model from: ', args.model_path)
    model = load_model(args.model_path)

    if args.convert_model:
        print('* Converting model softmax layer to linear ')
        model = convert_model(model)
    
    img_size = args.image_size
    imgs = visualize_cam(model, args.image_path, conv_layer = args.conv_layer_name, size = (img_size,img_size), hif = args.hif, bw = args.output_all)
    
    print('* Saving color heatmap at: ', output_path + '_heatmap_color.jpg')
    cv2.imwrite(output_path + '_heatmap_color.jpg', imgs[1])

    if args.output_all:
        print('* Saving original image at: ', output_path + '_original.jpg')
        cv2.imwrite(output_path + '_original.jpg', imgs[0])
        print('* Saving grayscale heatmap at: ', output_path + '_heatmap_bw.jpg')
        cv2.imwrite(output_path + '_heatmap_bw.jpg', imgs[2])






