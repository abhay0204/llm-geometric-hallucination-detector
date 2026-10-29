import json
import re
import os

INPUT_FILES = {"gpt": "results_gpt.txt", "gemini": "results_gemini.txt"}

def clean_text(text):
    return text.replace('**', '').strip()

def parse_file(filepath, source_name):
    if not os.path.exists(filepath):
        print(f"⚠️  Skipping {source_name}: File '{filepath}' not found.")
        return []

    print(f"📖 Processing {source_name} data from '{filepath}'...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(r'(?:\*\*|)?PROMPT #\d+\s+(?:—|-|--)\s+(.*?)_(factual|hallucinated)(?:\*\*|)?', re.IGNORECASE)
    chunks = re.split(r'(?:\*\*|)?PROMPT #\d+', content)
    
    entries = []
    headers = pattern.findall(content)
    data_chunks = chunks[1:] 

    for i, (rfc_id, type_label) in enumerate(headers):
        if i >= len(data_chunks): break
        
        raw_body = data_chunks[i]
        clean_lines = [line.strip() for line in raw_body.split('\n') if rfc_id not in line and "—" not in line and line.strip() != "---" and line.strip()]
        body_text = clean_text(" ".join(clean_lines))

        entries.append({
            "id": f"{rfc_id}_{type_label}",
            "type": type_label.lower(),
            "llm_output": body_text
        })
    return entries

if __name__ == "__main__":
    for name, filepath in INPUT_FILES.items():
        data = parse_file(filepath, name)
        if data:
            output_filename = f"{name}_results.jsonl"
            with open(output_filename, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry) + '\n')
            print(f"Saved {len(data)} entries to '{output_filename}'")