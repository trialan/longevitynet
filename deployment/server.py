from PIL import Image
from flask import Flask, request
import io
import numpy as np
import torch
import torchvision.transforms as transforms

from longevity.modelling.model import preprocess, ResNet50
from longevity.modelling.train import generate_dataset
from longevity.modelling.utils import undo_min_max_scaling

app = Flask(__name__)

model = ResNet50()
model.load_state_dict(torch.load('deployment/model.pth', map_location=torch.device('cpu')))
model.eval()
dataset = generate_dataset()


@app.route('/predict', methods=['POST'])
def predict():
    image = get_image(request)
    age_tensor = get_age(request)
    prediction = get_prediction(image, age_tensor)
    return {'life_expectancy': round(prediction, 2)}


def get_prediction(image, age_tensor):
    output = model(image.unsqueeze(0), age_tensor)
    raw_prediction = output.item()
    prediction = convert_to_years(raw_prediction)
    return prediction
    

def convert_to_years(raw_prediction):
    min_delta_value = np.min(dataset.deltas)
    max_delta_value = np.max(dataset.deltas)
    prediction = undo_min_max_scaling(raw_prediction,
                                      min_val = min_delta_value,
                                      max_val = max_delta_value)
    return prediction


def get_image(request):
    image = request.files['file']
    image = Image.open(io.BytesIO(image.read()))
    image = np.array(image)
    image = preprocess(image)
    return image


def get_age(request):
    age = float(request.form.get('age'))
    age_tensor = torch.tensor([age]).unsqueeze(0)
    return age_tensor


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)

