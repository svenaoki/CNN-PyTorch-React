from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import requests
from PIL import Image
import torchvision.transforms as transforms
from utils import convNet
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PATH = os.path.join(os.getcwd(), 'backend', 'python')


def load_model():
    model = convNet()
    model.load_state_dict(torch.load(os.path.join(
        PATH, 'state_dict_model.pt')))
    return model


def transform_image(image):
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((128, 128))])
    image = Image.open(image)
    return my_transforms(image).unsqueeze(0)


def get_prediction(image):
    model = load_model()
    model.eval()
    tensor = transform_image(image=image)
    outputs = model(tensor)
    sm = nn.Softmax()
    predictions = sm(outputs).detach().numpy()
    return np.around(predictions[0], decimals=4).tolist()


@app.route('/', methods=['POST'])
def predict():
    if request.files['file']:
        image = request.files['file']
        probs = get_prediction(image=image)
        return jsonify(probs)


if __name__ == "__main__":
    app.run(debug=True)
""" 
 # test
resp = requests.post("http://localhost:5000",
                     files={"file": open(os.path.join(os.getcwd(), 'dataset', 'test', 'dogs', 'dog.0.jpg'), 'rb')})
print(resp.json())
 """
