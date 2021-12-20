from flask import Flask, request, jsonify
import json
from _image_object_detection import reconocer_objeto, cargar_grafo
import base64
from cv2 import cv2
import numpy as np

app = Flask(__name__)


@app.route('/test_base_64', methods=['POST'])
def test_base_64():

    image = request.files['data'].read()
    image = np.asarray(bytearray(image), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    resultado = reconocer_objeto(image)

    return resultado


if __name__ == '__main__':
    cargar_grafo()
    app.run(debug=True, port=3001, host='0.0.0.0')
