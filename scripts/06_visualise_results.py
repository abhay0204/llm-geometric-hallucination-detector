import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
from importlib import import_module

detector_mod = import_module('04_run_detector')
HallucinationDetector = detector_mod.HallucinationDetector

INPUT_FILES = ["gemini_results.jsonl", "gpt_results.jsonl"]
OUTPUT_DIR = "dissertation_plots"

def generate_plots_for_file(input_file, detector):
    if not os.path.exists(input_file): return None

    model_name = input_file.replace("_results.jsonl", "").upper()
    print(f"\nProcessing {model_name} data from {input_file}...")
    
    facts, hallucinations, y_true, y_scores = [], [], [], []

    with open(input_file, 'r', encoding='utf-8') as f:
        for data in (json.loads(line) for line in f):
            text = data.get('llm_output', '')
            if not text: continue
            
            score = detector.get_anomaly_score(text)
            is_hall = 1 if 'hallucination' in data['type'].lower() or 'hallucinated' in data['type'].lower() else 0
            
            (hallucinations if is_hall else facts).append(score)
            y_true.append(is_hall)
            y_scores.append(score)

    if not facts or not hallucinations: return None

    # PLOT 1: HISTOGRAM
    plt.figure(figsize=(10, 6))
    sns.histplot(facts, color="green", label="Factual (Real)", kde=True, element="step", alpha=0.5)
    sns.histplot(hallucinations, color="red", label="Hallucinated (Fake)", kde=True, element="step", alpha=0.5)
    plt.title(f"Statistical Separation ({model_name})"); plt.xlabel("Euclidean Distance"); plt.ylabel("Frequency"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_1_histogram.png", dpi=300); plt.close()

    # PLOT 2: BOX PLOT
    plt.figure(figsize=(8, 6))
    plt.boxplot([facts, hallucinations], labels=['Factual', 'Hallucinated'], patch_artist=True, boxprops=dict(facecolor="lightblue"), medianprops=dict(color="red"))
    plt.title(f"Distribution Ranges - {model_name}"); plt.ylabel("Euclidean Distance"); plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_2_boxplot.png", dpi=300); plt.close()

    # PLOT 3: ROC CURVE
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (Area = {roc_auc:.2f})'); plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}'); plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_3_roc_curve.png", dpi=300); plt.close()
    
    return roc_auc

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    detector = HallucinationDetector()
    final_scores = {f: generate_plots_for_file(f, detector) for f in INPUT_FILES}
    
    print("\nFINAL VISUALIZATION REPORT")
    for fname, auc_score in final_scores.items():
        if auc_score: print(f"   {fname:<20} | AUROC: {auc_score:.4f}")