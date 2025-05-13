"""

This is a command-line tool for running image-based trademark search.
It loads CLIP and ResNet features, computes similarities, and prints ranked matches.

This isn't called in app.py as this was initially used to build and test, however and it's functionality and same pipeline is used in app.py

"""

import torch
import clip
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights

# --- Load models ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load CLIP ---
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# --- Load ResNet-50 ---
weights = ResNet50_Weights.DEFAULT
resnet_model = resnet50(weights=weights)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval().to(device)

# --- ResNet preprocessing ---
resnet_preprocess = weights.transforms()

# --- Load features and metadata ---
clip_feats = np.load("clip_features.npy")
resnet_feats = np.load("resnet_features.npy")
index_df = pd.read_csv("image_index.csv")
filenames = index_df["filename"].tolist()
mark_ids = index_df["mark_id"].tolist()
image_folder = Path("decoded_images")

# --- Hybrid search ---
def hybrid_search(text_query=None, query_image_path=None, top_k=5, alpha=0.5):
    clip_sim = np.zeros(len(filenames))
    resnet_sim = np.zeros(len(filenames))

    if text_query:
        text_tokens = clip.tokenize([text_query]).to(device)
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_tokens).cpu().numpy()
        clip_sim = cosine_similarity(text_feat, clip_feats).squeeze()

    if query_image_path:
        img = Image.open(query_image_path).convert("RGB")

        # --- CLIP features ---
        clip_tensor = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            query_clip_feat = clip_model.encode_image(clip_tensor).cpu().numpy()
        clip_sim = cosine_similarity(query_clip_feat, clip_feats).squeeze()

        #  ---ResNet features ---
        resnet_tensor = resnet_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            query_resnet_feat = resnet_model(resnet_tensor).cpu().numpy()
        resnet_sim = cosine_similarity(query_resnet_feat, resnet_feats).squeeze()

    # --- Combine similarity scores ---
    final_sim = alpha * clip_sim + (1 - alpha) * resnet_sim
    indices = final_sim.argsort()[::-1][:top_k]
    return [(filenames[i], mark_ids[i], final_sim[i]) for i in indices]

#--- Show results ---
def show_results(query_image_path, results, title="Hybrid Search Results"):
    plt.figure(figsize=(12, 3))
    if query_image_path:
        plt.subplot(1, len(results) + 1, 1)
        plt.imshow(Image.open(query_image_path))
        plt.title("Query")
        plt.axis("off")

    for i, (fname, mark_id, score) in enumerate(results):
        img = Image.open(image_folder / fname)
        plt.subplot(1, len(results) + 1, i + 2)
        plt.imshow(img)
        plt.title(f"ID: {mark_id}\n{score:.3f}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid trademark search with CLIP and ResNet.")
    parser.add_argument("--text", type=str, help="Text query (optional)")
    parser.add_argument("--image", type=str, help="Path to query image (optional)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for CLIP (0=ResNet only, 1=CLIP only)")
    parser.add_argument("--topk", type=int, default=5, help="Number of top matches to return")

    args = parser.parse_args()

    results = hybrid_search(
        text_query=args.text,
        query_image_path=args.image,
        alpha=args.alpha,
        top_k=args.topk
    )

    print("\nTop Matches:")
    for fname, mark_id, score in results:
        print(f"Mark ID: {mark_id} | File: {fname} | Score: {score:.3f}")

    if args.image:
        show_results(args.image, results)
