import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg19
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

def load_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device, torch.float)

class ContentLossModule(nn.Module):
    def __init__(self, target):
        super(ContentLossModule, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def compute_gram_matrix(input_tensor):
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLossModule(nn.Module):
    def __init__(self, target_feature):
        super(StyleLossModule, self).__init__()
        self.target = compute_gram_matrix(target_feature).detach()

    def forward(self, input):
        G = compute_gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    

# Load pre-trained VGG19 model
cnn = vgg19(pretrained=True).features.to(device).eval()

# Normalization mean and standard deviation
norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
