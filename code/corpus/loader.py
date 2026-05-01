import os
import re

def chunk_text(text, max_words=230, overlap_words=40):
    """
    Chunks text into roughly 300 token segments (approximated by 230 words)
    with a 50 token (40 word) overlap. Tries to respect paragraph boundaries.
    """
    # First, split by paragraphs to try to respect semantic boundaries
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for p in paragraphs:
        p_words = p.split()
        if not p_words:
            continue
            
        if current_length + len(p_words) <= max_words:
            current_chunk.extend(p_words)
            current_length += len(p_words)
        else:
            # Current paragraph pushes it over the limit
            if current_length > 0:
                chunks.append(" ".join(current_chunk))
                # Start new chunk with overlap from the previous
                # Take the last `overlap_words` from `current_chunk`
                overlap_text = current_chunk[-overlap_words:] if len(current_chunk) > overlap_words else current_chunk
                current_chunk = overlap_text + p_words
                current_length = len(current_chunk)
            else:
                # A single paragraph is larger than max_words, we must hard split it
                for i in range(0, len(p_words), max_words - overlap_words):
                    chunk_slice = p_words[i:i + max_words]
                    chunks.append(" ".join(chunk_slice))
                
                # Setup for the next iteration (empty it out since we processed the huge paragraph)
                current_chunk = []
                current_length = 0
                
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def load_corpus(data_dir):
    """
    Walks all files in data/{hackerrank,claude,visa}/, extracts text,
    and chunks into segments.
    Returns: List[dict(text, domain, source, chunk_id)]
    """
    domains = ["hackerrank", "claude", "visa"]
    corpus_data = []
    chunk_counter = 0
    
    for domain in domains:
        domain_dir = os.path.join(data_dir, domain)
        if not os.path.exists(domain_dir):
            print(f"Warning: Directory {domain_dir} does not exist.")
            continue
            
        for root, _, files in os.walk(domain_dir):
            for file in files:
                # Only process text-based files
                if file.endswith('.md') or file.endswith('.txt') or file.endswith('.html'):
                    file_path = os.path.join(root, file)
                    source = os.path.relpath(file_path, data_dir)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            
                        # Chunk the text
                        chunks = chunk_text(text)
                        
                        for c in chunks:
                            corpus_data.append({
                                "text": c,
                                "domain": domain,
                                "source": source,
                                "chunk_id": chunk_counter
                            })
                            chunk_counter += 1
                            
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        
    print(f"Loaded {len(corpus_data)} chunks from {data_dir}.")
    return corpus_data
