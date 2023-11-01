from PIL import Image
from flask import Flask, request
import io
import numpy as np
import torch
import torchvision.transforms as transforms

from longevitynet.modelling.model import preprocess, ResNet50
from longevitynet.modelling.utils import undo_min_max_scaling, get_gender_probs

app = Flask(__name__)

model = ResNet50()
model.load_state_dict(torch.load('longevitynet/deployment/best_model_mae_6p3.pth',
                                 map_location=torch.device('cpu')))
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    image = get_image(request)
    p_man, p_woman = get_gender_probs()
    age_tensor = get_age(request)
    prediction = get_prediction(image, age_tensor, p_man, p_woman)
    return {'longevitynet': round(prediction, 2)}


def get_prediction(image, age_tensor, p_man, p_woman):
    output = model(image.unsqueeze(0), age_tensor, p_man, p_woman)
    raw_prediction = output.item()
    prediction = convert_to_years(raw_prediction)
    return prediction


def get_gender_probs():
    p_man, p_woman = get_gender_probs("image.jpg")
    p_man = torch.tensor([[p_man]])
    p_woman = torch.tensor([[p_woman]])
    return p_man, p_woman

    

def convert_to_years(raw_prediction):
    return raw_prediction


def get_image(request):
    image = request.files['file']
    image = Image.open(io.BytesIO(image.read()))
    with open("image.jpg", "w") as f:
        f.write(image.read())
    image = np.array(image)
    image = preprocess(image)
    return image


def get_age(request):
    age = float(request.form.get('age'))
    age_tensor = torch.tensor([age]).unsqueeze(0)
    return age_tensor


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)

