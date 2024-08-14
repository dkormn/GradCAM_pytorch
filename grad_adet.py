# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:53:01 2024

@author: MONSTER
"""
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import os
import cv2
from PIL import Image
import random
from sklearn.metrics import jaccard_score, f1_score

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name())

# Veri transformasyonları
data_transforms = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

# Test klasöründeki tüm görüntü dosyalarını al
test_dir = r'C:\Users\MONSTER\PycharmProjects\pneumothorax_mask\Classification1\test\diseased'
image_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]

# 100 rastgele görüntü seç
random.seed(42)
selected_images = random.sample(image_files, 100)

# Modeli oluşturma ve eğitilmiş ağırlıkları yükleme
def CNN_Model():
   model = models.resnet34(pretrained=False)
   model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features=model.fc.in_features, out_features=2)
    )
   return model

model = CNN_Model()
checkpoint_path = r"C:\Users\MONSTER\PycharmProjects\pneumothorax_mask\checkpoints\best_resnet34.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# Grad-CAM sınıfı
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach().to(device)

    def save_gradients(self, module, input, output):
        self.gradients = output[0].detach().to(device)

    def generate(self, input_image, target_class=None):
        self.model.eval()
        input_image = input_image.to(device)

        output = self.model(input_image)

        if target_class is None:
            target_class = np.argmax(output.cpu().data.numpy())
        
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3))[0, :]
        
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i, :, :]
        
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cv2.resize(cam.cpu().numpy(), (input_image.shape[2], input_image.shape[3]))
        return cam

# Grad-CAM uygulama
gradcam = GradCAM(model, model.layer4[-1])

# Eşik değerler
iou_threshold = 0.0637
dice_threshold = 0.1066

correct_iou_count = 0
correct_dice_count = 0

# Seçilen 100 görüntü üzerinde işlem yap
for img_name in selected_images:
    image_path = os.path.join(test_dir, img_name)
    mask_path = os.path.join(r'C:\Users\MONSTER\PycharmProjects\pneumothorax_mask\siim-acr-pneumothorax\masks', img_name)
    
    if os.path.exists(mask_path):
        image = Image.open(image_path).convert('RGB')
        input_image = data_transforms(image).unsqueeze(0).to(device)
        
        mask = Image.open(mask_path).convert('L')
        mask = data_transforms(mask).squeeze(0).cpu().numpy()

        # Grad-CAM uygulayın
        cam_mask = gradcam.generate(input_image)
        
        # İkili eşikleme (thresholding)
        threshold = 0.5
        cam_mask_binary = (cam_mask > threshold).astype(np.uint8)
        mask_binary = (mask > threshold).astype(np.uint8)

        # IoU (Intersection over Union) Hesaplama
        iou = jaccard_score(mask_binary.flatten(), cam_mask_binary.flatten())

        # Dice Skoru (F1 Skoru) Hesaplama
        dice = f1_score(mask_binary.flatten(), cam_mask_binary.flatten())

        # Eşik değer kontrolü
        if iou > iou_threshold:
            correct_iou_count += 1
        if dice > dice_threshold:
            correct_dice_count += 1

    else:
        print(f"Mask not found for image: {img_name}. Skipping...")

# Sonuçları yazdır
print(f"{correct_iou_count} out of 100 images have IoU > {iou_threshold}")
print(f"{correct_dice_count} out of 100 images have Dice Score > {dice_threshold}")
