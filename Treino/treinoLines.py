import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from glob import glob

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

# ================= UNET =================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# ================= CONFIG MASK =================
def generate_mask_from_image(img, config, warp_size=(128, 128)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([config["HUE Min"], config["SAT Min"], config["VALUE Min"]])
    upper = np.array([config["HUE Max"], config["SAT Max"], config["VALUE Max"]])
    mask = cv2.inRange(hsv, lower, upper)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    wT = 640
    pts_src = np.float32([
        [config["Width Top"] + config["Shift Left"], config["Height Top"]],
        [wT - config["Width Top"] + config["Shift Left"], config["Height Top"]],
        [config["Width Bottom"] + config["Shift Right"], config["Height Bottom"]],
        [wT - config["Width Bottom"] + config["Shift Right"], config["Height Bottom"]]
    ])
    pts_dst = np.float32([[0, 0], [warp_size[0], 0], [0, warp_size[1]], [warp_size[0], warp_size[1]]])

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(mask_bgr, matrix, warp_size)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return warped_gray  # sem crop! usas a imagem toda


# ================= DATASET =================
class LineDataset(Dataset):
    def __init__(self, image_paths, config_path, img_size=(128, 128)):
        self.image_paths = image_paths
        self.img_size = img_size
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERRO] Imagem inválida: {img_path}")
            return None

        try:
            img = cv2.resize(img, (640, 480))
            mask = generate_mask_from_image(img, self.config, warp_size=self.img_size)
            if mask is None or mask.size == 0:
                print(f"[ERRO] Máscara inválida: {img_path}")
                mask = np.zeros(self.img_size, dtype=np.uint8)

            mask = cv2.resize(mask, self.img_size)
            mask_tensor = torch.tensor((mask > 127).astype(np.float32)).unsqueeze(0)
            img_tensor = torch.tensor(cv2.resize(img, self.img_size).transpose(2, 0, 1) / 255.0, dtype=torch.float32)

            return img_tensor, mask_tensor

        except Exception as e:
            print(f"[FATAL] {img_path} falhou com erro: {e}")
            return None


# ================= CSV LOADER =================
def load_images_from_csv(dataset_dir="dataset"):
    image_paths = []
    for session in glob(os.path.join(dataset_dir, "session_*")):
        csv_path = os.path.join(session, "steering_data.csv")
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = os.path.join(session, row["image_path"])
            if os.path.exists(img_path):
                image_paths.append(img_path)
    print(f"[INFO] {len(image_paths)} imagens carregadas do CSV")
    return sorted(image_paths)

# ================= DEBUG =================
def debug_segmentation(model, dataset, device, n=5, save_dir="debug_outputs"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    for i in range(n):
        img, true_mask = dataset[i]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))[0][0].cpu().numpy()
        vis = np.hstack([
            (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
            np.stack([true_mask[0].numpy() * 255]*3, axis=2).astype(np.uint8),
            np.stack([(pred > 0.5) * 255]*3, axis=2).astype(np.uint8)
        ])
        out = os.path.join(save_dir, f"debug_{i}.png")
        cv2.imwrite(out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"[✔] {out}")

# ================= TREINO =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = load_images_from_csv("dataset")
    dataset = LineDataset(image_paths, "trackbar_config.json")
    #print("[INFO] A validar todas as imagens...")
    #image_paths = [p for p in image_paths if cv2.imread(p) is not None]
    #print(f"[INFO] Apas verificacoo: {len(image_paths)} ")

    train_ds, val_ds = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, drop_last=True)


    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    best_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, 21):
        model.train()
        total_loss = 0
	print(f" Epoch {epoch} ",end='\r')
        for i,(x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        train_losses.append(total_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        print(f"[Epoch {epoch}] Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            torch.save(model.state_dict(), "unet_lines.pth")
            print("[✔] Novo modelo salvo: unet_lines.pth")

    # Curva
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.grid(True)
    plt.title("Loss UNet")
    plt.savefig("loss_curve_unet.png")
    plt.show()

    # Debug
    debug_segmentation(model, val_ds, device)


