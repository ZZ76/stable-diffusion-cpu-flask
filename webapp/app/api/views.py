from flask import jsonify, request, current_app
from . import api
import time
import base64
import cv2
# from .... import generator
# from .... import generator_test

def test_image():
    with open('app/static/combat-agent.60cad2.png', 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('utf-8')

def generate_b64_image(np_img):
    _, img_b64 = cv2.imencode('.png', np_img)
    img_b64 = base64.b64encode(img_b64).decode('utf-8')
    return img_b64

@api.route('/stable_diffusion_generate', methods=['GET'])
def stable_diffusion():
    width = int(request.args.get('width'))
    height = int(request.args.get('height'))
    steps = int(request.args.get('steps'))
    text = request.args.get('text')
    # image_b64 = test_image()
    image_b64 = generate_b64_image(current_app.generator.generate(width, height, text, steps))
    response = jsonify({'width': width, 'height': height, 'text': text, 'steps':steps, 'image': image_b64})
    return response
