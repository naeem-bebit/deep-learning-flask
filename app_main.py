from flask import Flask, render_template, request

from keras.preprocessing.image import load_image, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    imagefile.save(image_path = "./images/" + imagefile.filename)
    return render_template('index.html', prediction = classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
