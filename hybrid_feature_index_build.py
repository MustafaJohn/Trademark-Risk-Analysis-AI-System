"""

This script builds the CLIP and ResNet image feature index for the trademark dataset.
It saves embeddings and metadata for use in image similarity searches in app.py

"""

import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights


device = "cuda" if torch.cuda.is_available() else "cpu"

# --- CLIP model ---
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# --- ResNet-50 ---
weights = ResNet50_Weights.DEFAULT
resnet_model = resnet50(weights=weights)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1]) 
resnet_model.eval().to(device)

# --- Preprocessing using models recommend transforms ---
resnet_preprocess = weights.transforms()

# --- Image folder ---
image_folder = Path("decoded_images")
image_paths = sorted(image_folder.glob("*.png"))  

# --- Storage ---
clip_feats, resnet_feats, filenames, mark_ids = [], [], [], []

with torch.no_grad():
    for path in tqdm(image_paths, desc="Processing images"):
        try:
            img = Image.open(path).convert("RGB")

            # --- CLIP features ---
            clip_tensor = clip_preprocess(img).unsqueeze(0).to(device)
            clip_feat = clip_model.encode_image(clip_tensor).squeeze().cpu().numpy()

            #--- ResNet features --
            resnet_tensor = resnet_preprocess(img).unsqueeze(0).to(device)
            resnet_feat = resnet_model(resnet_tensor).squeeze().cpu().numpy()

            clip_feats.append(clip_feat)
            resnet_feats.append(resnet_feat)
            filenames.append(path.name)
            mark_ids.append(path.stem)  
        except Exception as e:
            print(f"❌ Error with {path.name}: {e}")

# --- Save features and index ---
np.save("clip_features.npy", np.array(clip_feats))
np.save("resnet_features.npy", np.array(resnet_feats))
pd.DataFrame({"filename": filenames, "mark_id": mark_ids}).to_csv("image_index.csv", index=False)

print("✅ Hybrid feature index built and saved.")
