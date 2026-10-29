import json
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import roc_auc_score
from sentence_transformers import CrossEncoder, SentenceTransformer

# --- CONFIGURATION ---
CORPUS_FILE = "rfc_corpus.jsonl"
RESULT_FILES = ["gemini_results.jsonl", "gpt_results.jsonl"]
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-small'
EMBEDDING_PATH = "corpus_embeddings.npy"
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'

def get_jaccard_distance(text1, text2):
    """Calculates 1 minus the word overlap (distance)."""
    set1, set2 = set(re.findall(r'\b\w+\b', text1.lower())), set(re.findall(r'\b\w+\b', text2.lower()))
    if not set1 or not set2: return 1.0
    return 1.0 - (len(set1.intersection(set2)) / len(set1.union(set2)))

if __name__ == "__main__":
    print("Loading Corpus & Models for Baselines...")
    
    # 1 Load Text Corpus
    corpus_texts, ref_data = [], {}
    if os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                corpus_texts.append(data['text'])
                ref_data[data['doc_id']] = data['text']
    else: 
        print(f"Error: {CORPUS_FILE} not found.")
        exit()

    # 2 Setup TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    corpus_tfidf = tfidf_vectorizer.fit_transform(corpus_texts)
    
    # 3 Setup NLI Cross-Encoder
    nli_model = CrossEncoder(NLI_MODEL_NAME, device='cpu')
    
    # 4 Setup SBERT (for Cosine calculations)
    if not os.path.exists(EMBEDDING_PATH):
        print(f"Error: {EMBEDDING_PATH} not found. Run vectorize_corpus.py first.")
        exit()
    
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    ref_embeddings = np.load(EMBEDDING_PATH)
    ref_norms = np.linalg.norm(ref_embeddings, axis=1)

    def get_cosine_min_distance(target_text):
        """Calculates the minimum Cosine Distance (1 - Cosine Similarity) to the corpus."""
        target_vec = sbert_model.encode([target_text])[0]
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0: return 1.0
        
        dot_products = np.dot(ref_embeddings, target_vec)
        similarities = dot_products / (ref_norms * target_norm)
        return np.min(1 - similarities)

    print("\nRunning All Benchmarks...\n")

    for filename in RESULT_FILES:
        if not os.path.exists(filename): continue

        jaccard_scores, tfidf_scores, cosine_scores = [], [], []
        nli_pairs, labels, nli_labels = [], [], []
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data.get('llm_output', '')
                if not text: continue

                is_hall = 1 if 'hallucinated' in data['type'].lower() else 0
                labels.append(is_hall)

                # --- Calculate Distances ---
                jaccard_scores.append(min([get_jaccard_distance(text, ref) for ref in corpus_texts]))
                tfidf_scores.append(min(cosine_distances(tfidf_vectorizer.transform([text]), corpus_tfidf)[0]))
                cosine_scores.append(get_cosine_min_distance(text))

                # --- NLI Setup ---
                core_id = "_".join(data['id'].split('_')[:-1])
                if core_id in ref_data:
                    nli_pairs.append([ref_data[core_id], text])
                    nli_labels.append(is_hall)

        print(f"Results for {filename}:")
        print(f"-> Lexical (Jaccard) AUROC:      {roc_auc_score(labels, jaccard_scores):.4f}")
        print(f"-> Classical ML (TF-IDF) AUROC:  {roc_auc_score(labels, tfidf_scores):.4f}")
        
        # Calculate Cosine metrics + The Separation Margin
        cosine_auc = roc_auc_score(labels, cosine_scores)
        fact_scores = [s for s, l in zip(cosine_scores, labels) if l == 0]
        hall_scores = [s for s, l in zip(cosine_scores, labels) if l == 1]
        margin = np.mean(hall_scores) - np.mean(fact_scores)
        
        print(f"-> SBERT (Cosine) AUROC:  {cosine_auc:.4f} (Margin: {margin:.4f})")
        
        if nli_pairs:
            contradiction_scores = nli_model.predict(nli_pairs)[:, 0]
            print(f"-> Neural Semantic (NLI) AUROC:  {roc_auc_score(nli_labels, contradiction_scores):.4f}\n")