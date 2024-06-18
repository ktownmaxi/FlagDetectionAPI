import base64
import os.path

from flask import Flask, request
from flask_restful import Api

import eval_img

app = Flask(__name__)
api = Api(app)


def predict_on_image(img_path):
    prediction_instance = eval_img.FlagPredictionBuilder()
    prediction = prediction_instance.make_prediction(img_path)
    return prediction


@app.route('/analyzeImage', methods=['POST'])
def process_image():
    if 'image' not in request.json:
        return 'No image provided', 400

    encoded_image = request.json['image']
    decoded_image = base64.b64decode(encoded_image)
    save_path = os.path.join("uploaded_images", "image.png")
    with open(save_path, 'wb') as f:
        f.write(decoded_image)

    final_prediction = predict_on_image(save_path)

    final_string = {'status': 'ok',
                    'prediction': final_prediction}

    return final_string, 200


if __name__ == '__main__':
    app.run(host="0.0.0.0")
