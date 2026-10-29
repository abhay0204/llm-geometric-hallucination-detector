# Geometric Anomaly Detection for LLM Hallucinations

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![AUROC: 0.953](https://img.shields.io/badge/AUROC-0.953%20(Gemini)-orange.svg)]()
[![Hardware: CPU Only](https://img.shields.io/badge/Inference-Sub--millisecond%20CPU-brightgreen.svg)]()

> **Undergraduate Final Year Dissertation (2026)**  
> **Author:** Abhay Praseeth — *BSc Computer Science & Artificial Intelligence, Loughborough University*  
> **Paper:** [Read Full Dissertation PDF](paper/dissertation.pdf)

An offline, zero-token, black-box geometric anomaly detection pipeline designed to intercept plausible technical hallucinations in commercial Large Language Models (GPT-4 and Gemini Pro) without auxiliary LLM calls, external search engines, or white-box parameter access.

---

## The Core Problem: The Black-Box Dilemma

Mitigating hallucinations in production typically relies on two unworkable paradigms:
1. **Retrieval-Augmented Generation (RAG):** Demands external database lookups, adds 2,000–4,000ms latency, and fails in air-gapped environments.
2. **Internal State Analysis (White-Box):** Inspects hidden-layer activations or token log-probabilities—telemetry that commercial API providers (OpenAI, Google) do not expose.

### The Hypothesis: Geometry as a Proxy for Truth
In rigid technical domains (e.g., IETF Internet Standards/RFCs), factual text exhibits a predictable statistical density manifold. Rather than attempting neural reading comprehension, this framework treats hallucination detection as a **geometric anomaly problem**: mapping model outputs into a 384-dimensional latent semantic space (`all-MiniLM-L6-v2`) and calculating the $k$-Nearest Neighbour ($k=1$) Euclidean distance against an indexed manifold of 9,727 ground-truth RFC paragraph chunks.

```
[20 IETF RFCs] ──► 00_rfc_to_jsonl.py ──► [9,727 Ground-Truth Chunks]
                                                    │
[Adversarial Prompts] ──► 01_gen_prompts.py          ▼
                                           03_vectorise_corpus.py
                                                    │ (all-MiniLM-L6-v2)
                                                    ▼
[GPT-4 / Gemini Outputs] ───────────────► 04_run_detector.py ──► AUROC: 0.953
                                                    ▲
[Jaccard / TF-IDF / NLI] ───────────────► 05_run_baselines.py ──► AUROC: 0.874
                                                    │
                                          06_visualise_results.py ──► [Plots & ROC Curves]
```

---

## The Key Discovery: "The Euclidean Pivot"

Standard NLP literature defaults to **Cosine Similarity**. However, experiments revealed that **Cosine Similarity suffers severe threshold compression** when evaluating technical fabrications:

* **Why Cosine Similarity Fails:** When an LLM fabricates a technical specification (e.g., inventing an HTTP status code or header rule), it heavily recycles valid domain terminology. This keeps the output vector pointing in the same angular direction as genuine documentation (yielding a deceptive cosine similarity near 1.0).
* **Why Euclidean Distance Succeeds:** While the angular direction is preserved, the lack of hyper-specific lexical density displaces the vector outward. **Euclidean distance ($L_2$ norm) captures the physical spatial magnitude of this drift.**

| Metric | Model | Factual Mean Baseline | Hallucinated Mean | Separation Margin (Gap) |
| :--- | :--- | :--- | :--- | :--- |
| **Cosine Distance** | Gemini Pro | 0.812 | 0.985 | **0.1733** (Compressed) |
| **Euclidean Distance ($L_2$)** | **Gemini Pro** | **0.528** | **0.797** | **0.2690** (+55.2% Wider Margin) |
| **Cosine Distance** | GPT-4 | 0.831 | 0.976 | **0.1453** (Compressed) |
| **Euclidean Distance ($L_2$)** | **GPT-4** | **0.562** | **0.783** | **0.2215** (+52.4% Wider Margin) |

> Switching to Euclidean distance widened the statistical separation margin by over 50%, pulling hallucinated clusters safely away from the ground-truth baseline to enable reliable operational thresholding without false positives.

---

## Benchmark Results

Evaluated across an adversarial dataset of paired factual summaries and academic exam distractor fabrications:

| Detection Paradigm | Evaluator Architecture | Gemini Pro AUROC | GPT-4 AUROC | Operational Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **Lexical Overlap** | Jaccard Similarity | 0.9026 | 0.9086 | $O(N)$ String Match |
| **Sparse Vector ML** | TF-IDF + Cosine Distance | 0.8916 | 0.9260 | Penalizes valid synonyms |
| **Neural Reading (NLI)** | `cross-encoder/nli-deberta-v3-small` | 0.8735 | 0.8742 | $O(N)$ Heavy Cross-Attention |
| **Geometric (Ours)** | **SBERT ($k=1$) Euclidean Distance** | **0.9532** | **0.9276** | **O(1) Sub-millisecond CPU** |

### Key Empirical Findings:
1. **Outperforming DeBERTa Cross-Encoders:** As the reference manifold scaled to 9,727 chunks, the NLI model suffered severe degradation (~0.87 AUROC) due to overlapping technical jargon across protocols. The geometric pipeline maintained ~0.95 AUROC because spatial magnitude is immune to semantic reading confusion.
2. **The Vocabulary Paradox:** Jensen-Shannon Divergence (JSD) testing revealed that hallucinated outputs exhibited **lower perplexity** than authentic RFC texts. LLMs fall back on common, fluent vocabulary when inventing technical facts, meaning "smoother" technical text is statistically more likely to be fabricated.
3. **Hardware Independence:** All vector calculations execute in $<1$ ms on a standard Apple Silicon M1 CPU (8GB RAM) with zero GPU acceleration.

---

## Repository Structure

```text
llm-geometric-hallucination-detector/
├── .gitignore
├── requirements.txt
├── README.md
├── LICENSE
├── paper/
│   └── Abhay_Praseeth_Dissertation_2026.pdf      # Complete 37-page research thesis
├── data/
│   ├── rfc_corpus/                              # Raw RFC source texts
│   ├── rfc_corpus.jsonl                         # 9,727 extracted paragraph chunks
│   ├── llm_prompts.jsonl                        # Paired factual & adversarial prompts
│   ├── gemini_results.jsonl                     # Raw Gemini Pro model generations
│   └── gpt_results.jsonl                        # Raw GPT-4 model generations
├── plots/                                       # Publication-ready figures
│   ├── roc_curve_gemini.png
│   ├── roc_curve_gpt.png
│   ├── separation_histograms.png
│   └── distribution_boxplots.png
├── scripts/
│   ├── 00_rfc_to_jsonl.py                       # Ingests RFCs and parses into JSONL
│   ├── 01_gen_prompts.py                        # Generates paired exam distractor prompts
│   ├── 02_parse_results.py                      # Cleans and normalizes LLM outputs
│   ├── 03_vectorise_corpus.py                   # Embeds 9,727 chunks via MiniLM-L6-v2
│   ├── 04_run_detector.py                       # Computes k=1 Euclidean vs Cosine scores
│   ├── 05_run_baselines.py                      # Evaluates Jaccard, TF-IDF, & DeBERTa NLI
│   └── 06_visualise_results.py                  # Generates ROC curves & boxplots
└── outputs/
    ├── results_gemini.txt
    └── results_gpt.txt
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/abhay0204/llm-geometric-hallucination-detector.git
cd llm-geometric-hallucination-detector
pip install -r requirements.txt
```

### 2. Reproduce Detection & Baselines
```bash
# Run the geometric detector over Gemini & GPT outputs
python scripts/04_run_detector.py

# Run comparative baselines (Jaccard, TF-IDF, DeBERTa NLI)
python scripts/05_run_baselines.py

# Re-generate ROC curves, separation histograms, and boxplots
python scripts/06_visualise_results.py
```

---

## Citation

```bibtex
@thesis{praseeth2026geometric,
  author       = {Abhay Praseeth},
  title        = {Security Checks for Large Language Models: A Statistical Approach to Hallucination Detection},
  school       = {Loughborough University},
  year         = {2026},
  type         = {BSc Dissertation},
  note         = {Grade: 2:1}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
