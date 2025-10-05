import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

import numpy as np 
import streamlit as st
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import gdown
import cv2

file_id = "1GrkMfHTY6-kqOthmWEYHVWYkPcoqCCkR"
output_path = "best_unet_oilspill.pth"

if not os.path.exists(output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# --- UNet Model Architecture ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load(output_path, map_location=device))
model.eval()

# --- Helper: Clean mask using morphology ---
def clean_mask(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
    mask_clean = (mask_clean > 127).astype(np.uint8)
    return mask_clean

# --- Helper: Overlay mask on image with color ---
def overlay_image_color(img, mask, color=(255,0,0), alpha=0.4):
    img = img.resize(mask.shape[::-1])
    img_np = np.array(img).astype(np.uint8)
    mask = mask.astype(bool)
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    overlay = img_np.copy()
    overlay[mask] = color
    blended = cv2.addWeighted(img_np, 1-alpha, overlay, alpha, 0)
    return Image.fromarray(blended)

st.title("Oil Spill Segmentation Demo (U-Net)")
st.write("Upload a satellite image to detect oil spill area.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def predict_mask(pil_img, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output)
        pred_mask = (prob > 0.5).float().cpu().numpy()[0, 0]
    return pred_mask

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width='stretch')
    mask = predict_mask(img)
    mask_clean = clean_mask(mask)
    st.image(mask_clean, caption="Cleaned Predicted Mask", width='stretch', clamp=True)

    # Oil spill alert
    spill_pixels = np.sum(mask_clean == 1)
    total_pixels = mask_clean.size
    spill_percent = (spill_pixels / total_pixels) * 100
    THRESHOLD = 2.0

    if spill_percent > THRESHOLD:
        st.success(f"⚠️ Oil spill detected! ({spill_percent:.2f}% of area)")
    else:
        st.info(f"No significant oil spill detected. ({spill_percent:.2f}% of area)")

    overlay = overlay_image_color(img, mask_clean, color=(255,0,0), alpha=0.4)
    st.image(overlay, caption="Oil Spill Overlay (Red)", width='stretch')

    # Download mask
    mask_img = Image.fromarray((mask_clean * 255).astype(np.uint8))
    buf = BytesIO()
    mask_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Mask", data=byte_im, file_name="mask.png", mime="image/png")

st.markdown("---")
st.write("Model by KhushiBadsra | Powered by Streamlit + PyTorch")
