'''

This Python script is the main file for running the front end.
It defines the Streamlit UI and connects to functions in ocr_functions and db_utils to perform text and image-based trademark search and generate downloadable reports.

Dependencies:
Access to Database
First run marks_trademarks_images.py
Then hybrid_feature_index_build.py

'''


import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import torch
import clip
import pdfkit
import tempfile
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import resnet50, ResNet50_Weights
from ocr_functions import extract_text_from_image, load_semantic_index, combined_search
from db_utils import fetch_image_by_trademark_id
from io import BytesIO
import base64
import zipfile
from io import BytesIO

# --- Convert PIl Image to Base64 String ---
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- Load the models and precomputed data---
@st.cache_resource
def load_models_and_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    weights = ResNet50_Weights.DEFAULT
    resnet_model = resnet50(weights=weights)
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
    resnet_model.eval().to(device)

    resnet_preprocess = weights.transforms()
    clip_feats = np.load("clip_features.npy")
    resnet_feats = np.load("resnet_features.npy")
    df_index = pd.read_csv("image_index.csv")
    filenames = df_index["filename"].tolist()
    mark_ids = df_index["mark_id"].tolist()

    return clip_model, clip_preprocess, resnet_model, resnet_preprocess, clip_feats, resnet_feats, filenames, mark_ids, device

# --- Hybrid Image similarity Function ---
def hybrid_search(image_path, clip_model, clip_preprocess, resnet_model, resnet_preprocess,
                  clip_feats, resnet_feats, filenames, mark_ids, device, alpha=0.5, top_k=5):
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        clip_tensor = clip_preprocess(img).unsqueeze(0).to(device)
        clip_feat = clip_model.encode_image(clip_tensor).cpu().numpy()

        resnet_tensor = resnet_preprocess(img).unsqueeze(0).to(device)
        resnet_feat = resnet_model(resnet_tensor).cpu().numpy().reshape(1, -1)

    clip_sim = cosine_similarity(clip_feat, clip_feats).squeeze()
    resnet_sim = cosine_similarity(resnet_feat, resnet_feats).squeeze()
    final_sim = alpha * clip_sim + (1 - alpha) * resnet_sim

    indices = final_sim.argsort()[::-1][:top_k]
    return [(filenames[i], mark_ids[i], final_sim[i]) for i in indices]

# --- Func to generate PDF from results ---
def create_pdf(semantic_df, visual_list, image_path, filename="trademark_report.pdf"):
    semantic_rows = ""
    for _, row in semantic_df.iterrows():
        semantic_rows += f"""
        <tr>
            <td>{row['trademark_id']}</td>
            <td>{row['verbal_element_text']}</td>
            <td>{row['CombinedScore']:.4f}</td>
        </tr>
        """

    visual_rows = ""
    for fname, mark_id, score in visual_list:
        img_path = Path("decoded_images") / fname
        image_uri = img_path.resolve().as_uri()
        visual_rows += f"""
        <tr>
            <td><img src=\"{image_uri}\" width=\"100\"></td>
            <td>{mark_id}</td>
            <td>{fname}</td>
            <td>{score:.4f}</td>
        </tr>
        """

    html = f"""
    <html><body>
    <h1>Trademark Search Report</h1>
    <h2>Query Image</h2>
    <img src="{Path(image_path).resolve().as_uri()}" width="200"><br><br>

    <h2>Semantic OCR Search Results</h2>
    <table border="1" cellpadding="8">
        <tr><th>Trademark ID</th><th>Text</th><th>Combined Score</th></tr>
        {semantic_rows}
    </table>

    <h2>Visual Similarity Results (CLIP + ResNet)</h2>
    <table border="1" cellpadding="8">
        <tr><th>Image</th><th>Mark ID</th><th>Filename</th><th>Similarity</th></tr>
        {visual_rows}
    </table>
    </body></html>
    """
    options = {"enable-local-file-access": None}
    pdf_bytes=pdfkit.from_string(html, False, options=options)
    return pdf_bytes

# --- Streamlit App ---
st.set_page_config(page_title="The Trademark House Hybrid Search", layout="centered")
st.title("The Trademark House Hybrid Search Engine")
clip_model, clip_preprocess, resnet_model, resnet_preprocess, clip_feats, resnet_feats, filenames, mark_ids, device = load_models_and_data()

# --- Mode switching and reset ---
current_mode = st.radio("Choose Search Mode", ["📁 Upload Image", "✏️ Type Description"])
previous_mode = st.session_state.get("previous_mode")
if previous_mode and previous_mode != current_mode:
    for key in ["ocr_text", "semantic_df", "visual_results", "desc_semantic_df", "desc_clip_results", "pdf_path"]:
        st.session_state.pop(key, None)
st.session_state["previous_mode"] = current_mode
search_mode = current_mode

top_k = st.slider("Number of Top Matches", 1, 10, 5)

if search_mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a trademark image", type=["jpg", "jpeg", "png"])
    alpha = st.slider("CLIP–ResNet Weight", 0.0, 1.0, 0.5, 0.05)

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if st.button("Run Image Search"):
            for key in ["ocr_text", "semantic_df", "visual_results", "desc_semantic_df", "desc_clip_results"]:
                st.session_state.pop(key, None)

            ocr_text = extract_text_from_image(tmp_path)
            st.session_state["ocr_text"] = ocr_text
            st.session_state["image_path"] = tmp_path

            df_sem, sem_embeddings = load_semantic_index()
            semantic_df = combined_search(ocr_text, df_sem, sem_embeddings, top_k=top_k)
            st.session_state["semantic_df"] = semantic_df

            visual_results = hybrid_search(tmp_path, clip_model, clip_preprocess, resnet_model,
                                           resnet_preprocess, clip_feats, resnet_feats,
                                           filenames, mark_ids, device, alpha, top_k)
            st.session_state["visual_results"] = visual_results
            st.success("✅ Image search completed!")

elif search_mode == "✏️ Type Description":
    typed_description = st.text_input("Enter a textual description:", placeholder="e.g. a red star on a circle")

    if typed_description and st.button("Search by Description"):
        for key in ["ocr_text", "semantic_df", "visual_results", "desc_semantic_df", "desc_clip_results"]:
            st.session_state.pop(key, None)

        df_sem, sem_embeddings = load_semantic_index()
        desc_semantic_df = combined_search(typed_description, df_sem, sem_embeddings, top_k=top_k)
        st.session_state["desc_semantic_df"] = desc_semantic_df

        with torch.no_grad():
            text_features = clip_model.encode_text(clip.tokenize([typed_description]).to(device)).cpu().detach().numpy()
        clip_sim = cosine_similarity(text_features, clip_feats).squeeze()
        top_indices = clip_sim.argsort()[::-1][:top_k]
        clip_results = [(filenames[i], mark_ids[i], clip_sim[i]) for i in top_indices]
        st.session_state["desc_clip_results"] = clip_results
        st.success("✅ Text search completed!")

# --- Results ---
if "ocr_text" in st.session_state:
    st.markdown("### OCR Extracted Text")
    st.write(f"`{st.session_state['ocr_text']}`")

if "semantic_df" in st.session_state:
    st.markdown("### Semantic Matches (from OCR)")
    df_display = st.session_state["semantic_df"][["trademark_id", "verbal_element_text", "CombinedScore", "Semantic", "Fuzzy", "Phonetic"]].copy()
    df_display.index = np.arange(1, len(df_display) + 1)
    st.dataframe(df_display, use_container_width=True)
    st.markdown("### Matching Images from Database (from OCR search)")
    for _, row in st.session_state["semantic_df"].iterrows():
        trademark_id = int(row["trademark_id"])
        img = fetch_image_by_trademark_id(trademark_id)
        if img:
            st.image(img, caption=f"Trademark ID: {trademark_id}", width=150)
    

if "visual_results" in st.session_state:
    st.markdown("### Visual Matches (CLIP + ResNet)")
    for fname, mark_id, score in st.session_state["visual_results"]:
        st.image(Path("decoded_images") / fname, width=150)
        st.write(f"**Mark ID**: `{mark_id}` — **File**: `{fname}` — **Similarity**: `{score:.4f}`")

if "desc_semantic_df" in st.session_state:
    st.markdown("### Semantic Matches (from Description)")
    df_desc = st.session_state["desc_semantic_df"][["trademark_id", "verbal_element_text", "CombinedScore", "Semantic", "Fuzzy", "Phonetic"]].copy()
    df_desc.index = np.arange(1, len(df_desc) + 1)
    st.dataframe(df_desc, use_container_width=True)
    st.markdown("### Matching Images from Database (from OCR search)")
    for _, row in st.session_state["desc_semantic_df"].iterrows():
        trademark_id = int(row["trademark_id"])
        img = fetch_image_by_trademark_id(trademark_id)
        if img:
            st.image(img, caption=f"Trademark ID: {trademark_id}", width=150)

if "desc_clip_results" in st.session_state:
    st.markdown("### Visual Matches (from Description)")
    for fname, mark_id, score in st.session_state["desc_clip_results"]:
        st.image(Path("decoded_images") / fname, width=150)
        st.write(f"**Mark ID**: `{mark_id}` — **File**: `{fname}` — **Similarity**: `{score:.4f}`")

# --- PDF Export ----
if all(key in st.session_state for key in ["semantic_df", "visual_results", "image_path"]):
    if st.button("Generate PDF Report"):
        pdf_bytes = create_pdf(
            st.session_state["semantic_df"],
            st.session_state["visual_results"],
            st.session_state["image_path"]
        )
        st.session_state["pdf_bytes"] = pdf_bytes
        st.success("✅ PDF report generated!")

if "pdf_bytes" in st.session_state:
    st.download_button(
        label="Download PDF Report",
        data=st.session_state["pdf_bytes"],
        file_name="trademark_report.pdf",
        mime="application/pdf"
    )


# --- Clear All Button ---
if st.button("Clear All Results"):
    for key in ["ocr_text", "semantic_df", "visual_results", "desc_semantic_df", "desc_clip_results", "pdf_path"]:
        st.session_state.pop(key, None)
    st.success("All results cleared.")
