import torch
from PIL import Image
import open_clip
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.init as init
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json 
import numpy as np
from torchvision.transforms import InterpolationMode
from ultralytics import YOLO
from PIL import Image, ImageDraw



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model,_, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
print(vit_model)

def smooth_labels(binary_labels, sigma):
    # Define the Gaussian kernel
    size = 2 * sigma + 1
    kernel = torch.exp(-(torch.arange(size) - sigma)**2 / (2 * sigma**2)).to(device)
    kernel = kernel / kernel.sum()
    
    padded_labels = F.pad(binary_labels.unsqueeze(0), (0, 0))
    kernel = kernel.view(1, 1, -1)
    
    smoothed_labels = F.conv1d(padded_labels.unsqueeze(0), kernel, padding=sigma)[0][0]
    return smoothed_labels

def loss_function(predicted_probs, labels):
    loss = nn.BCELoss()
    return loss(predicted_probs, labels)

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
image_transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

class MLPProjector(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPProjector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        nn.init.kaiming_uniform_(self.fc1.weight, a=0.01, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2.weight, a=0.01, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

   

class SelfAttentionBlock(nn.Module):
    def __init__(self, input_size, num_heads=8):
        super(SelfAttentionBlock, self).__init__()
        self.normlayer = nn.LayerNorm(normalized_shape=(input_size,))
        self.self_attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, 4 * input_size),
            nn.GELU(),
            nn.Linear(4 * input_size, input_size)
        )
        
    def forward(self, x):
        residual = x
        out = self.normlayer(x)
        out, _ = self.self_attention(out, out, out)
        out += residual
        residual = out
        out = self.normlayer(out)
        out = self.feedforward(out)
        out += residual
        return out
    

class KeyFrameExtractor_v4(nn.Module):
    """
    a relatively easy model with 
    """
    def __init__(self, num_classes=10,  num_layers=2):
        super(KeyFrameExtractor_v4, self).__init__()
        self.clip_encoder = vit_model
        
        for param in self.clip_encoder.parameters():
            param.requires_grad = False
            
        self.attention_layers = nn.ModuleList([
            SelfAttentionBlock(input_size=256) for _ in range(num_layers)
        ])
        self.position_embedding = nn.Parameter(torch.randn(10, 1, 256)) 
        self.mlp_projector = MLPProjector(input_size=512, hidden_size=512*4, output_size=256)
        self.normlayer=  nn.LayerNorm(normalized_shape=(10,))
       
    def forward(self, images):
        flattened_images = [self.clip_encoder.encode_image(im.unsqueeze(0)) for image in images for im in image]
        features = torch.stack(flattened_images) 
        projected_features = self.mlp_projector(features) 
        projected_features += self.position_embedding
        out= projected_features
        for layer in self.attention_layers:
            out= layer(out) 
             
        out = out.permute(1,0,2)
        # print(out.size())
        out = out.mean(dim=2) # (B, 10)
       
        return out


class Cursor_detector:
    def __init__(self, check_point_path, video_dir):
        super(Cursor_detector, self).__init__()
        self.detection_model = YOLO(check_point_path)
        self.video_dir = video_dir
        
    def detect(self):
        for j in range(10):
            image_path= f'{self.video_dir}/frame_{j}.png'
            results = self.detection_model(image_path)
            img = Image.open(image_path)
            width, height = img.size
            img.close()
            print(width, height)
            for result in results:
                if result.boxes.xywh.size(0)>0:
                    boxes = result.boxes 
                    xywh_tensor = boxes.xywh
                    x, y = xywh_tensor[0][0].item(),xywh_tensor[0][1].item()
                    # print("Value of the first tensor:", x,y)
                    image1 = Image.open(image_path).convert('RGB') 
                    x1, y1= max(0, x-128), max(0, y-128)
                    start_crop = image1.crop((x1, y1, min(x1 + 256,width), min(y1 + 256,height)))
                    start_crop.save(self.video_dir+f'/{j}_crop.png')
                    x1 = max(0, x - 128)
                    y1 = max(0, y - 128)
                    x2 = min(x1 + 256, width)
                    y2 = min(y1 + 256, height)
                    
                    # Draw the bounding box on the image
                    draw = ImageDraw.Draw(image1)
                    draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
                    image1.save(self.video_dir+f'/{j}_prompt.png')
                    image1.close()
                else:
                    image1 = Image.open(image_path).convert('RGB') 
                    x1, y1= max(0, x-128), max(0, y-128)
                    start_crop = image1.crop((x1, y1, min(x1 + 256,width), min(y1 + 256,height)))
                    start_crop.save(self.video_dir+f'/{j}_crop.png')
                    x1 = max(0, x - 128)
                    y1 = max(0, y - 128)
                    x2 = min(x1 + 256, width)
                    y2 = min(y1 + 256, height)
                    draw = ImageDraw.Draw(image1)
                    draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
                    image1.save(self.video_dir+f'/{j}_prompt.png')
                    image1.close()
        
class ImageReader:
    def __init__(self, root_dir, transform=image_transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for i in range(10): 
                image_path = os.path.join(self.root_dir, f'{i}_crop.png')
                if os.path.exists(image_path):
                    image_paths.append(image_path)
        return image_paths

    def read_images(self):
        images = []
        for image_path in self.image_paths:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return torch.stack(images)

class VideoReader:
    pass

