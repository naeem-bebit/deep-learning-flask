from flask import Flask, render_template, request, jsonify

# from keras.preprocessing.image import load_image, img_to_array
# from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

# import tensorflow as tf
# # from tensorflow import keras

# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET'])
def predict():
#     # imagefile = request.files['imagefile']
#     # imagefile.save(image_path = "./images/" + imagefile.filename)
#     # return render_template('index.html', prediction = classification)
    # return jsonify({'result':1})
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
