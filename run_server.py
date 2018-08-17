# import keras
import tensorflow as tf

from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.utils.data_utils import get_file
from keras.models import load_model

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import flask
import io
import os
import re
import json
import quality_detection

app = flask.Flask(__name__)
model = None

img_net_file = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

fpath = get_file('imagenet_class_index.json',
                 img_net_file,
                 cache_subdir='models',
                 file_hash='c2c37ea517e94d9795004a39431a14cb')
with open(fpath) as f:
    CLASS_INDEX = json.load(f)
    
pattern = re.compile(r'clock|watch')
clock_mask = np.array([bool(pattern.search(v[1])) for v in CLASS_INDEX.values()])
# clock_names = [CLASS_INDEX[str(i)][1] for i, c in enumerate(clock_mask) if c]


def get_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
#     model = ResNet50(weights="imagenet")
    model = load_model('model_watch.h5')
    global graph
    graph = tf.get_default_graph()
    
    
def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

def image2resized_array(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    return image

def imagenet_format(image):
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

def process_clock_proba(predict_proba, pos_mask):
    # this function is used in the older version with imagenet 1000 classification output
    positive_proba = np.sum(predict_proba[0,pos_mask])
    if positive_proba >= np.max(predict_proba[0,~pos_mask]):
        prediction = (1, positive_proba)
    else:
        prediction = (0, 1-positive_proba)
    return prediction

@app.route("/")
def index():
    return "hey there"

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            quality_score = quality_detection.image_quality_detection(image)            

            image = image2resized_array(image, target=(224, 224))
            image = imagenet_format(image)
            
            with graph.as_default():
                preds = model.predict(image)

                preds = model.predict(image)[0][0]
            pred_label = 1 if preds>=0.5 else 0

            r = {"label": pred_label, "probability": round(np.float(preds),3), \
         "quality": quality_score}
                
            data["predictions"] = r
        return flask.jsonify(data["predictions"])

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    get_model()
    app.run(host='0.0.0.0')