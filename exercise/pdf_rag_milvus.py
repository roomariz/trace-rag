import hashlib
import os
import re
import requests
from pypdf import PdfReader
from pymilvus import MilvusClient


def compute_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_chunk_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


url = "https://arxiv.org/pdf/1706.03762.pdf"
pdf_path = "sample.pdf"

if not os.path.exists(pdf_path):
    print("PDF not found, downloading...")
    response = requests.get(url)
    with open(pdf_path, "wb") as f:
        f.write(response.content)

current_hash = compute_file_hash(pdf_path)
print(f"Current PDF hash: {current_hash[:16]}...")


def extract_sections(reader):
    sections = []
    current_section = {"heading": "Introduction", "text": "", "page": 0}
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if re.match(r"^(#+\s+|CHAPTER\s+|SECTION\s+|[0-9]+\.\s+[A-Z])", line.upper()):
                if current_section["text"]:
                    sections.append(current_section)
                current_section = {"heading": line, "text": "", "page": page_num}
            else:
                current_section["text"] += " " + line
        
        current_section["page"] = page_num
    
    if current_section["text"]:
        sections.append(current_section)
    
    return sections


def chunk_text(text, chunk_size=500, overlap=50):
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    current = ""
    start_idx = 0

    for i, s in enumerate(sentences):
        if len(current) + len(s) < chunk_size:
            current += " " + s
        else:
            if current:
                chunks.append({
                    "text": current.strip(),
                    "start_idx": start_idx,
                    "end_idx": i
                })
            current = s
            start_idx = i

    if current:
        chunks.append({
            "text": current.strip(),
            "start_idx": start_idx,
            "end_idx": len(sentences)
        })

    return chunks


def chunk_text_with_metadata(text, chunk_size=500):
    section_pattern = r'^\d+\s+[A-Z]'
    subsection_pattern = r'^\d+\.\d+\s+[A-Z]'
    
    chunks = []
    current_section = ""
    current_subsection = ""
    current_chunk = ""
    paragraph_id = 0
    
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if re.match(section_pattern, line):
            current_section = line
            current_subsection = ""
        elif re.match(subsection_pattern, line):
            current_subsection = line
        else:
            if len(current_chunk) + len(line) < chunk_size:
                current_chunk += " " + line
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": 0,
                        "section": current_section,
                        "subsection": current_subsection,
                        "paragraph_id": paragraph_id
                    })
                    paragraph_id += 1
                current_chunk = line
    
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "page": 0,
            "section": current_section,
            "subsection": current_subsection,
            "paragraph_id": paragraph_id
        })
    
    return chunks


def embed(texts):
    vectors = []

    for text in texts:
        res = requests.post(
            "http://localhost:11435/api/embed",
            json={
                "model": "mxbai-embed-large",
                "input": [text]
            }
        )
        data = res.json()
        vectors.append(data["embeddings"][0])

    return vectors


client = MilvusClient(uri="tcp://localhost:19530")

collection_exists = client.has_collection("pdf_collection")

stored_hash = None
if collection_exists:
    try:
        results = client.query(
            collection_name="pdf_collection",
            filter="id == -1",
            output_fields=["text"]
        )
        if results and "text" in results[0]:
            stored_hash = results[0]["text"]
    except:
        pass

if stored_hash == current_hash:
    print("PDF and embeddings unchanged, skipping ingestion")
else:
    if collection_exists:
        print("PDF updated, dropping old embeddings")
        client.drop_collection("pdf_collection")
    collection_exists = False

if not collection_exists:
    print("Ingesting PDF...")

    reader = PdfReader(pdf_path)

    section_pattern = r'^\d+\s+[A-Z]'
    subsection_pattern = r'^\d+\.\d+\s+[A-Z]'
    
    chunks = []
    current_section = ""
    current_subsection = ""
    current_chunk = ""
    paragraph_id = 0
    
    print("Detecting sections...")
    
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text:
            continue
        
        lines = page_text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            is_section_1 = re.match(r'^\d+\s+\w', line) and len(line) < 80
            is_section_2 = re.match(r'^\d+\.\d+\s+\w', line) and len(line) < 80
            is_section_3 = re.match(r'^\d+\.\d+\.\d+\s+\w', line) and len(line) < 80
            
            if is_section_1:
                current_section = line
                current_subsection = ""
            elif is_section_3:
                current_subsection = line
            elif is_section_2:
                current_subsection = line
                print(f"Section: {line[:40]}")
            else:
                if len(current_chunk) + len(line) < 500:
                    current_chunk += " " + line
                else:
                    if current_chunk:
                        para_id = paragraph_id
                        chunks.append({
                            "text": current_chunk.strip(),
                            "page": page_num + 1,
                            "section": current_section,
                            "subsection": current_subsection,
                            "paragraph_id": para_id
                        })
                        print(f"  -> Chunk {len(chunks)}: sec={current_section[:20] if current_section else 'none'}")
                        paragraph_id += 1
                    current_chunk = line
    
    if current_chunk:
        para_id = paragraph_id
        chunks.append({
            "text": current_chunk.strip(),
            "page": page_num + 1,
            "section": current_section,
            "subsection": current_subsection,
            "paragraph_id": para_id
        })
        print(f"  -> Last Chunk: sec={current_section[:20] if current_section else 'none'}")
    
    print(f"Loaded PDF text length: {sum(len(c['text']) for c in chunks)}")
    print(f"Number of chunks: {len(chunks)}")

    dim = len(embed([chunks[0]["text"]])[0])

    client.create_collection(
        collection_name="pdf_collection",
        dimension=dim,
        metric_type="COSINE"
    )

    all_texts = [c["text"] for c in chunks]
    vectors = embed(all_texts)

    data = [
        {
            "id": i,
            "vector": vectors[i],
            "text": c["text"],
            "page": c.get("page", 0),
            "section": c.get("section", ""),
            "subsection": c.get("subsection", ""),
            "paragraph_id": c.get("paragraph_id", 0),
            "chunk_hash": compute_chunk_hash(c["text"])
        }
        for i, c in enumerate(chunks)
    ]

    client.insert(collection_name="pdf_collection", data=data)

    client.insert(
        collection_name="pdf_collection",
        data=[{"id": -1, "vector": [0.0] * dim, "text": current_hash}]
    )

    client.flush("pdf_collection")
    client.load_collection("pdf_collection")

    print("PDF embeddings stored")

query = "What is attention in transformers?"

SYNTHESIS_KEYWORDS = ["why", "problem", "compare", "advantage", "difference", "how does", "benefits"]
is_synthesis = any(kw in query.lower() for kw in SYNTHESIS_KEYWORDS)

INITIAL_LIMIT = 15 if is_synthesis else 10

query_vector = embed([query])

MAX_CONTEXT_CHARS = 2500 if is_synthesis else 2000

results = client.search(
    collection_name="pdf_collection",
    data=query_vector,
    limit=INITIAL_LIMIT,
    output_fields=["id", "text", "heading", "page"],
)

retrieved = results[0]
print(f"Retrieved {len(retrieved)} chunks")

candidates = []
for hit in retrieved:
    candidates.append({
        "text": hit["entity"]["text"],
        "score": hit["distance"],
        "id": hit["id"]
    })

candidate_texts = [c["text"] for c in candidates]
candidate_vectors = embed(candidate_texts)

reranked = []
for i, c in enumerate(candidates):
    sim = sum(a * b for a, b in zip(query_vector[0], candidate_vectors[i]))
    c["rerank_score"] = sim
    reranked.append(c)

reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

scores = [c["rerank_score"] for c in reranked]
if len(scores) >= 3:
    score_spread = scores[0] - scores[-1]
    top3_spread = scores[0] - scores[2]
    
    if top3_spread < 0.05:
        QUERY_MARGIN = 0.05
    elif score_spread > 0.25:
        QUERY_MARGIN = 0.02
    else:
        QUERY_MARGIN = 0.1
else:
    QUERY_MARGIN = 0.1

MAX_NEIGHBOURS_PER_HIT = 1 if is_synthesis else 2

seen = set()
unique_hits = []
total_chars = 0

for c in reranked:
    if total_chars + len(c["text"]) > MAX_CONTEXT_CHARS:
        continue
    if c["text"] not in seen:
        unique_hits.append(c)
        seen.add(c["text"])
        total_chars += len(c["text"])

BASE_MARGIN = 0.15 if is_synthesis else 0.1
BASE_NEIGHBOURS = 1 if is_synthesis else 2

expanded_context = []
context_order = []

scores = [c["rerank_score"] for c in reranked]
if len(scores) >= 3:
    score_spread = scores[0] - scores[-1]
    top3_spread = scores[0] - scores[2]
    
    if top3_spread < 0.05:
        QUERY_MARGIN = BASE_MARGIN - 0.05
    elif score_spread > 0.25:
        QUERY_MARGIN = BASE_MARGIN + 0.05
    else:
        QUERY_MARGIN = BASE_MARGIN
else:
    QUERY_MARGIN = BASE_MARGIN

MAX_NEIGHBOURS_PER_HIT = BASE_NEIGHBOURS

top_score = reranked[0]["rerank_score"] if reranked else 0.0
min_neighbour_score = top_score - QUERY_MARGIN

for hit in unique_hits:
    hit_id = hit["id"]
    hit_text = hit["text"]
    
    expanded_context.append(hit_text)
    context_order.append(("main", hit["rerank_score"], hit_text))
    seen.add(hit_text)
    
    neighbours = client.query(
        collection_name="pdf_collection",
        filter=f"id >= {hit_id - 1} and id <= {hit_id + 1}",
        output_fields=["text"],
        limit=3
    )
    
    if neighbours:
        neighbor_texts = [n["text"] for n in neighbours]
        neighbor_vectors = embed(neighbor_texts)
        hit_vector = embed([hit_text])[0]
        
        neighbour_count = 0
        for i, n in enumerate(neighbours):
            if neighbour_count >= MAX_NEIGHBOURS_PER_HIT:
                break
            if n["text"] in seen or total_chars >= MAX_CONTEXT_CHARS:
                continue
            
            sim_to_query = sum(a * b for a, b in zip(query_vector[0], neighbor_vectors[i]))
            sim_to_main = sum(a * b for a, b in zip(hit_vector, neighbor_vectors[i]))
            
            if sim_to_query >= min_neighbour_score and sim_to_main >= min_neighbour_score:
                expanded_context.append(n["text"])
                context_order.append(("expand", sim_to_main, n["text"]))
                seen.add(n["text"])
                total_chars += len(n["text"])
                neighbour_count += 1

context = "\n\n".join(expanded_context)

print(f"Context size: {len(context)} chars")

for hit in unique_hits:
    print("\n--- Result ---")
    print("Score:", hit["rerank_score"])
    print(hit["text"][:300])

prompt = f"""
You are an expert AI assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
)

print("\n--- Answer ---")
print(response.json()["response"])