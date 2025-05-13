"""

This script provides functions for OCR-based text extraction and hybrid text similarity search.
It uses easyOCR, Sentence-BERT, RapidFuzz, and Double Metaphone to compute semantic, lexical,and phonetic similarity scores between trademarks.

"""


import easyocr
import cv2
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from rapidfuzz import fuzz
from metaphone import doublemetaphone

# ---  Load OCR reader and sentence transformer --- 
reader = easyocr.Reader(['en'], gpu=True)
model = SentenceTransformer('all-MiniLM-L6-v2')  

JUNK_WORDS = ['limited', 'ltd', 'inc', 'plc', 'group', 'company', 'co', 'uk', 'llp']

# ---  Preprocess image --- 
def preprocess_image_for_ocr(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return img

# ---  Extract OCR text --- 
def extract_text_from_image(image_path):
    preprocessed = preprocess_image_for_ocr(image_path)
    result = reader.readtext(preprocessed, detail=0)
    return " ".join(result)

#  --- Load text data and embed  --- 
def load_semantic_index(csv_path="trademark_images_export.csv"):
    df = pd.read_csv(csv_path)
    df["verbal_element_text"] = df["verbal_element_text"].fillna("").astype(str)
    embeddings = model.encode(df["verbal_element_text"].tolist(), convert_to_tensor=True)
    return df, embeddings

# ---  Helper to clean names --- 
def clean_name(name):
    tokens = name.lower().replace('.', '').split()
    return ' '.join([t for t in tokens if t not in JUNK_WORDS])

# --- Phonetic score using metaphone --- 
def soundslike_score(name1, name2):
    mp1 = doublemetaphone(name1)
    mp2 = doublemetaphone(name2)
    scores = [fuzz.ratio(c1, c2) for c1 in mp1 for c2 in mp2 if c1 and c2]
    return max(scores) / 10 if scores else 0

# --- Combined semantic, fuzzy, and phonetic search --- 
def combined_search(query_text, df, embeddings, top_k=5, w_semantic=0.6, w_fuzzy=0.3, w_phonetic=0.1):
    query_clean = clean_name(query_text)
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    semantic_scores = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()

    fuzzy_scores = df["verbal_element_text"].apply(lambda x: fuzz.token_sort_ratio(query_clean, clean_name(x)) / 100).values
    phonetic_scores = df["verbal_element_text"].apply(lambda x: soundslike_score(query_text, x)).values

    combined = w_semantic * semantic_scores + w_fuzzy * fuzzy_scores + w_phonetic * phonetic_scores
    indices = np.argsort(combined)[::-1][:top_k]

    results = df.iloc[indices].copy()
    results["CombinedScore"] = combined[indices]
    results["Semantic"] = semantic_scores[indices]
    results["Fuzzy"] = fuzzy_scores[indices]
    results["Phonetic"] = phonetic_scores[indices]
    results.reset_index(drop=True, inplace=True)
    return results
