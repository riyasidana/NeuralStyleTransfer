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


class NormalizationModule(nn.Module):
    def __init__(self, mean, std):
        super(NormalizationModule, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
    
# Specify layers for content and style
content_layers = ['conv_3']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def create_model_and_losses(cnn, norm_mean, norm_std, style_img, content_img):
    normalization = NormalizationModule(norm_mean, norm_std).to(device)
    content_loss_layers = []
    style_loss_layers = []

    model = nn.Sequential(normalization)
    conv_counter = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv_counter += 1
            name = f'conv_{conv_counter}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{conv_counter}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{conv_counter}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{conv_counter}'
        else:
            raise ValueError(f'Unknown layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLossModule(target)
            model.add_module(f"content_loss_{conv_counter}", content_loss)
            content_loss_layers.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLossModule(target_feature)
            model.add_module(f"style_loss_{conv_counter}", style_loss)
            style_loss_layers.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLossModule) or isinstance(model[i], StyleLossModule):
            break

    model = model[:i+1]
    return model, style_loss_layers, content_loss_layers
