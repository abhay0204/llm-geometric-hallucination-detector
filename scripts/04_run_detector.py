import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import jensenshannon
from collections import Counter

class HallucinationDetector:
    def __init__(self, corpus_path="rfc_corpus.jsonl", embedding_path="corpus_embeddings.npy", model_name='all-MiniLM-L6-v2'):
        print("\nINITIALIZING DETECTOR ENGINE...")
        self.model = SentenceTransformer(model_name)
        
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"ERROR: '{embedding_path}' not found.")
        self.ref_embeddings = np.load(embedding_path)
        print(f"- Loaded Reference Manifold: {self.ref_embeddings.shape[0]} vectors.")
        
        self.vocab_dist, self.total_words = Counter(), 0
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = json.loads(line)['text'].lower().split() 
                self.vocab_dist.update(tokens)
                self.total_words += len(tokens)
        print("- Detector Ready.\n")

    def get_anomaly_score(self, target_text):
        target_vec = self.model.encode([target_text])[0]
        dists = np.linalg.norm(self.ref_embeddings - target_vec, axis=1)
        return np.min(dists)

    def evaluate_batch(self, input_file):
        print(f"STARTING BATCH EVALUATION: {input_file}")
        if not os.path.exists(input_file):
            print(f"Error: '{input_file}' not found."); return

        entries = [json.loads(line) for line in open(input_file, 'r', encoding='utf-8') if json.loads(line).get('llm_output')]
        analysis = {"euclidean_min": {"fact": [], "hall": []}}
        
        for entry in entries:
            text = entry['llm_output']
            key = "fact" if "factual" in entry['type'].lower() else "hall"
            analysis["euclidean_min"][key].append(self.get_anomaly_score(text))

        print(f"\n{'METRIC':<15} | {'FACTUAL':<8} | {'HALLUC':<8} | {'MARGIN':<8}")
        print("-" * 50)
        
        data = analysis["euclidean_min"]
        avg_f, avg_h = np.mean(data['fact']), np.mean(data['hall'])
        margin = avg_h - avg_f
        status = "✓" if margin > 0.05 else "X"
        print(f"{'Euclidean Min':<15} | {avg_f:.4f}   | {avg_h:.4f}   | {margin:.4f}   | {status}")

if __name__ == "__main__":
    detector = HallucinationDetector()
    for filename in ["gemini_results.jsonl", "gpt_results.jsonl"]:
        if os.path.exists(filename):
            print("="*60)
            detector.evaluate_batch(filename)
            print("="*60 + "\n")