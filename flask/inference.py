import json
from commons import get_tensor, get_model
from torch import nn

mydict = {0:'Great! Your X-ray seems Normal', 1: 'Please, proceed to further examinations ! Pneumonia Symptoms :('}

model = get_model()


def get_xray_predict(image_bytes):
    tensor = get_tensor(image_bytes)
    print(tensor.shape)
    outputs = model.forward(tensor)
    print(outputs)
    Sof = nn.Softmax(dim=1)
    proba = Sof(outputs)[0][1]
    _, y_hat = outputs.max(1)

    return mydict[y_hat.item()] 


