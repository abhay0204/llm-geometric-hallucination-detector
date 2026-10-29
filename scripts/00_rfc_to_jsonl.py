import os
import json
import re

# --- CONFIGURATION ---
RAW_RFC_DIR = "rfc_corpus"
OUTPUT_FILE = "rfc_corpus.jsonl"

def clean_text(text):
    """
    Strips the boilerplate RFC formatting and reconstructs broken sentences.
    """
    # 1 Remove Form Feeds (\f) used for page breaks
    text = text.replace('\f', '')
    
    # 2 Remove page markers
    text = re.sub(r'\[Page \d+\]', '', text)
    
    # 3 Remove standard RFC headers/footers
    text = re.sub(r'RFC \d+.*?\n', '', text)
    
    # 4 Remove author/category footers
    # Usually characterized by large blocks of spaces
    text = re.sub(r'\w+\s{5,}.*?\n', '', text)
    
    # 5 Fix hard-wrapped lines 
    text = text.replace('\n', ' ')
    
    # 6 Collapse multiple spaces into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_rfcs():
    if not os.path.exists(RAW_RFC_DIR):
        print(f"Error: Folder '{RAW_RFC_DIR}' not found.")
        return

    print("Initializing RFC to JSONL Converter...")
    
    total_chunks = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_file:
        # Sort files to process them in a neat, predictable order
        for filename in sorted(os.listdir(RAW_RFC_DIR)):
            if not filename.endswith(".txt"):
                continue
                
            # Extract title (e.g., "rfc768.txt" -> "RFC768")
            rfc_title = filename.replace('.txt', '').upper()
            filepath = os.path.join(RAW_RFC_DIR, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                
            # --- CHUNKING STRATEGY ---
            # RFCs separate actual paragraphs using double newlines.
            paragraphs = raw_content.split('\n\n')
            
            chunk_index = 1
            for para in paragraphs:
                cleaned_para = clean_text(para)
                
                # --- FILTERING STRATEGY ---
                # We want to ignore empty chunks, short section headers, or Table of Contents dots (.....)
                if len(cleaned_para) > 50 and "....." not in cleaned_para:
                    
                    # Construct the JSON object
                    entry = {
                        "doc_id": f"{rfc_title}_{chunk_index}",
                        "title": rfc_title,
                        "text": cleaned_para
                    }
                    
                    out_file.write(json.dumps(entry) + '\n')
                    
                    total_chunks += 1
                
                # We always increment the chunk_index so that the IDs align sequentially with the physical structure of the document, even if we skip bad paragraphs.
                chunk_index += 1
                    
            print(f"Processed {rfc_title}")
            
    print(f"\nSUCCESS: Extracted {total_chunks} clean paragraphs into '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    process_rfcs()