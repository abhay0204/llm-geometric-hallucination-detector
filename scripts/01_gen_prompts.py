import json
import random
import os

# --- CONFIGURATION ---
INPUT_CORPUS = "rfc_corpus.jsonl"
OUTPUT_PROMPTS = "llm_prompts.jsonl"
SAMPLE_SIZE = 50 # 50 = 50+50 100 Prompts

if not os.path.exists(INPUT_CORPUS):
    print(f"Error: '{INPUT_CORPUS}' not found. run corpus converter first.")
    exit()

print(f"Reading {INPUT_CORPUS}...")
with open(INPUT_CORPUS, 'r', encoding='utf-8') as f:
    corpus_data = [json.loads(line) for line in f]

# Filter for chunks that are long enough
long_chunks = [d for d in corpus_data if len(d['text']) > 200]
print(f"   Found {len(long_chunks)} suitable chunks in the specific RFCs.")

selected_chunks = long_chunks if len(long_chunks) < SAMPLE_SIZE else random.sample(long_chunks, SAMPLE_SIZE)
prompts_to_save = []

print(f"Generating prompts for {len(selected_chunks)} chunks...")
for chunk in selected_chunks:
    doc_id, base_text, title = chunk['doc_id'], chunk['text'], chunk['title']
    
    # FACTUAL
    prompts_to_save.append({
        "id": f"{doc_id}_factual",
        "type": "factual",
        "reference_text": base_text,
        "prompt": f"You are a technical network engineer writing documentation. \nHere is a standard text from {title}:\n\"{base_text}\"\n\nSummarize this text in exactly one sentence. Be strictly factual and do not add outside information."
    })
    
    # HALLUCINATION
    prompts_to_save.append({
        "id": f"{doc_id}_hallucinated",
        "type": "hallucination",
        "reference_text": base_text,
        "prompt": f"You are writing a difficult exam for network engineers. \nHere is a standard text from {title}:\n\"{base_text}\"\n\nTask: Create a 'distractor' answer (a plausible but incorrect statement) based on this text.\nWrite a paragraph that *looks* like a real technical explanation but includes a completely invented header, error code, or protocol rule that does not actually exist.\nDo not preface it with \"Here is the incorrect answer.\" Just write the technical paragraph as if it were fact."
    })

with open(OUTPUT_PROMPTS, 'w', encoding='utf-8') as f:
    for entry in prompts_to_save:
        f.write(json.dumps(entry) + '\n')

print(f"\n Generated {len(prompts_to_save)} aligned prompts in '{OUTPUT_PROMPTS}'.")