import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import io

def get_model():
    checkpoint_path = 'final_Model.pt'

    input_vector = 128*30*30
    class Rob(nn.Module):
        def __init__(self):
            super(Rob, self).__init__()
            # convolutional layer
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 128, 3, padding=1)
            
            # MLP layers
            self.fc1 = nn.Linear(input_vector, 256)
            self.fc2 = nn.Linear(256,512)
            self.fc3 = nn.Linear(512,512)
            self.fc4 = nn.Linear(512,64)
            self.fc5 = nn.Linear(64,2)
            
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            # add sequence of convolutional and max pooling layers
            x = self.pool(F.relu(self.conv1(x)))
            x = F.relu(self.conv2(x))
            x = self.pool(F.relu(self.conv3(x)))
            
            # Flattening image to MPL
            x = x.view(-1, input_vector)
            
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            
            x = F.relu(self.fc3(x))
            x = self.dropout(x)

            x = F.relu(self.fc4(x))
            x = self.dropout(x)

            x = F.relu(self.fc5(x))
            
            return x

    model = Rob()
   
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model

def get_tensor(image_bytes):
    myTransforms = transforms.Compose([transforms.Resize((120,120)), 
                                        transforms.ToTensor()])
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return myTransforms(image).unsqueeze(0)
    