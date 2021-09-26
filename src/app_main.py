from flask import Flask, render_template, request, jsonify
import sys
import numpy as np
# from util import base64_to_pil

import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
model = MobileNetV2(weights='imagenet')

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds

# @app.route('/predict', methods = ['POST'])
@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    # Make prediction
    preds = model_predict(file_path, model)

    # Process your result for human
    # pred_class = preds.argmax(axis=-1)            # Simple argmax
    pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
    result = str(pred_class[0][0][1])               # Convert to string
    return result
    # return None
#     # imagefile = request.files['imagefile']
#     # imagefile.save(image_path = "./images/" + imagefile.filename)
#     # return render_template('index.html', prediction = classification)
    # return jsonify({'result':1})
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
