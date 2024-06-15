import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from code.imagelab import final_imagelab_model
from torchvision import transforms

model=final_imagelab_model(in_channels=3)
model_path="./model_23.pth"
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully")

input_dir="./test/low/"
output_dir="./test/predicted/"
for pngimage in os.listdir(input_dir):
    input_path = os.path.join(input_dir,pngimage)
    output_path = os.path.join(output_dir,pngimage)
    image = Image.open(input_path).convert('RGB')
    input_tensor=transform(image)
    input_tensor=input_tensor.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor=input_tensor.to(device)
    input_tensor=input_tensor.squeeze(1)
    enhanced=model(input_tensor)
    output_image=enhanced.squeeze(0).cpu()
    output_image=transforms.ToPILImage()(output_image)
    output_image.save(output_path)