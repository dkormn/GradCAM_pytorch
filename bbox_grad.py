import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import random
import matplotlib.pyplot as plt

# Cihaz ayarı (GPU varsa kullanılır)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name())

# Veri transformasyonları
data_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

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
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach().to(device)

    def save_gradients(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        self.gradients = output.detach().to(device)

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

# Bounding Box (BBox) oluşturma
def get_bbox_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(contour) for contour in contours]
    return bboxes

def overlay_gradcam_on_image(image, cam, alpha=0.5):
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))  # Boyutları eşitle
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(image) / 255  # Görüntüyü normalize et
    overlay = overlay / np.max(overlay)
    return np.uint8(255 * overlay)

# Test klasöründeki tüm görüntü dosyalarını al
image_dir = r'C:\Users\MONSTER\PycharmProjects\pneumothorax_mask\Classification1\test\diseased'
mask_dir = r'C:\Users\MONSTER\PycharmProjects\pneumothorax_mask\siim-acr-pneumothorax\mask'
image_files = os.listdir(image_dir)

# Maskesi olan görüntüleri filtrele
mask_files = [f for f in image_files if os.path.exists(os.path.join(mask_dir, f))]
print(f"Toplam maskeli görüntü sayısı: {len(mask_files)}")

# 100 rastgele maskeli görüntü seç
random.seed(42)
selected_images = random.sample(mask_files, 10)

# Görüntüleri işle
for idx, img_name in enumerate(selected_images):
    print(f"Processing image {idx+1}/{len(selected_images)}: {img_name}")
    
    image_path = os.path.join(image_dir, img_name)
    mask_path = os.path.join(mask_dir, img_name)
    
    # Mask ve görüntüyü yükle
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        continue

    # Grad-CAM için giriş görüntüsünü hazırlama
    input_image = data_transforms(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    cam = GradCAM(model, model.layer4[-1]).generate(input_image)

    # CAM'i ve maskeyi orijinal boyutlara geri döndürme
    original_image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
    cam_resized = cv2.resize(cam, (mask.shape[1], mask.shape[0]))
    
    # Maskeyi orijinal görüntünün üzerine bindirme
    mask_colored = np.zeros_like(original_image)
    mask_colored[mask > 0] = [0, 255, 0]  # Maskeyi yeşil yapıyoruz
    masked_image = cv2.addWeighted(original_image, 1, mask_colored, 0.5, 0)

    # Maske için BBox çizimi
    for bbox in get_bbox_from_mask(mask):
        x, y, w, h = bbox
        cv2.rectangle(masked_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Yeşil BBox (Maske)
    
    # Grad-CAM için BBox çizimi
    cam_threshold = cam_resized >= 0.4  # Eşik değerini burada değiştirebilirsiniz
    cam_bboxes = get_bbox_from_mask(cam_threshold.astype(np.uint8))
    for bbox in cam_bboxes:
        x, y, w, h = bbox
        cv2.rectangle(masked_image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Kırmızı BBox (Grad-CAM)

    # Grad-CAM'i orijinal görüntü üzerine bindirme
    cam_overlay = overlay_gradcam_on_image(original_image, cam_resized)
    
    # Sonucu görselleştir
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Image {idx+1} with Mask (Green) and Grad-CAM (Red) BBox")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Görselleştirmesi")
    plt.axis('off')
    
    plt.show()
