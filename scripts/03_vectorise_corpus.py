import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

INPUT_CORPUS = "rfc_corpus.jsonl"
OUTPUT_EMBEDDINGS = "corpus_embeddings.npy"
MODEL_NAME = 'all-MiniLM-L6-v2' 

if not os.path.exists(INPUT_CORPUS):
    print(f"ERROR: '{INPUT_CORPUS}' not found. Run the converter script first.")
    exit()

print(f"Loading SBERT model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

print(f"Reading {INPUT_CORPUS}...")
texts = [json.loads(line)['text'] for line in open(INPUT_CORPUS, 'r', encoding='utf-8')]

print(f"Vectorizing {len(texts)} text chunks...")
embeddings = model.encode(texts, show_progress_bar=True)

print(f"Saving vectors to '{OUTPUT_EMBEDDINGS}'...")
np.save(OUTPUT_EMBEDDINGS, embeddings)

print("\nVerifying saved file...")
loaded_vecs = np.load(OUTPUT_EMBEDDINGS)
if loaded_vecs.shape == (len(texts), 384) and np.any(loaded_vecs[0]):
    print("Reference Corpus Vectorized & Verified.")
else:
    print("ERROR: Could not verify .npy file shape/data.")