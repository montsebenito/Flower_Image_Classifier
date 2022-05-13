import sys
import time 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import argparse
import json

batch_size = 32
image_size = 224

class_names = {}

def process_image(image): 
   
    image = tf.cast(image, tf.float32)
    image= tf.image.resize(image, (image_size, image_size)).numpy()
    image /= 255
    
    return image
    

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    test_image = np.asarray(image)
    processed_test_image = process_image(test_image)
    expanded_test_image = np.expand_dims(processed_test_image, axis=0)
    
    pred_image = model.predict(expanded_test_image)
    values, indices = tf.math.top_k(pred_image, k=top_k)
    probs = values.numpy()[0]
    classes = indices.numpy()[0] + 1
    
    probs = list(probs)
    classes = list(map(str, classes))
    
    return probs, classes    


if __name__ == '__main__':
       
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('--top_k')
    
    args = parser.parse_args()
    print(args)
    
    print('image_path:', args.image_path)
    print('top_k:', args.top_k)
    
    model = tf.keras.models.load_model('TrainedModel.h5' ,custom_objects={'KerasLayer':hub.KerasLayer}, compile=False )
    top_k = args.top_k
    if top_k is None: 
        top_k = 5

    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
   
    probs, classes = predict(args.image_path, model, top_k)
    
    print(probs)
    print([class_names[label] for label in classes])