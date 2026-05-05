import subprocess
import sys
import os
import re

test_queries = [
    "What is attention in transformers?",
    "What is self-attention?",
    "Why is multi-head attention used?",
    "What problem does the Transformer solve?",
    "How does encoder-decoder attention work?",
    "What is the difference between encoder and decoder?",
    "Why is the Transformer faster than RNNs?",
]

def run_query(query):
    import requests
    
    from pypdf import PdfReader
    from pymilvus import MilvusClient
    import hashlib
    
    def compute_file_hash(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def compute_chunk_hash(text):
        return hashlib.sha256(text.encode()).hexdigest()

    pdf_path = "sample.pdf"
    current_hash = compute_file_hash(pdf_path)

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
                    chunks.append({"text": current.strip(), "start_idx": start_idx, "end_idx": i})
                current = s
                start_idx = i
        if current:
            chunks.append({"text": current.strip(), "start_idx": start_idx, "end_idx": len(sentences)})
        return chunks

    def embed(texts):
        vectors = []
        for text in texts:
            res = requests.post(
                "http://localhost:11435/api/embed",
                json={"model": "mxbai-embed-large", "input": [text]}
            )
            data = res.json()
            vectors.append(data["embeddings"][0])
        return vectors

    client = MilvusClient(uri="tcp://localhost:19530")
    collection_exists = client.has_collection("pdf_collection")

    stored_hash = None
    if collection_exists:
        try:
            results = client.query(collection_name="pdf_collection", filter="id == -1", output_fields=["text"])
            if results and "text" in results[0]:
                stored_hash = results[0]["text"]
        except:
            pass

    if stored_hash != current_hash:
        return None, "PDF changed, need to re-run ingestion"

    query_vector = embed([query])

    SYNTHESIS_KEYWORDS = ["why", "problem", "compare", "advantage", "difference", "how does", "benefits"]
    is_synthesis = any(kw in query.lower() for kw in SYNTHESIS_KEYWORDS)

    INITIAL_LIMIT = 15 if is_synthesis else 10
    QUERY_MARGIN = 0.15 if is_synthesis else 0.1
    MAX_NEIGHBOURS_PER_HIT = 1 if is_synthesis else 2
    MAX_CONTEXT_CHARS = 2500 if is_synthesis else 2000

    results = client.search(
        collection_name="pdf_collection",
        data=query_vector,
        limit=INITIAL_LIMIT,
        output_fields=["id", "text"],
    )

    retrieved = results[0]
    candidates = [{"text": h["entity"]["text"], "score": h["distance"], "id": h["id"]} for h in retrieved]

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

    expanded_context = []
    top_score = reranked[0]["rerank_score"] if reranked else 0.0
    min_neighbour_score = top_score - QUERY_MARGIN

    for hit in unique_hits:
        hit_id = hit["id"]
        hit_text = hit["text"]

        expanded_context.append(hit_text)
        seen.add(hit_text)

        neighbours = client.query(
            collection_name="pdf_collection",
            filter=f"id >= {hit_id - 1} and id <= {hit_id + 1}",
            output_fields=["text"],
            limit=3
        )

        if neighbours:
            neighbor_texts = [n["text"] for n in neighbours]
            neighbor_vectors = embed(neighbor_texts) if neighbor_texts else []
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
                    seen.add(n["text"])
                    total_chars += len(n["text"])
                    neighbour_count += 1

    context = "\n\n".join(expanded_context)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": f"""Answer using ONLY the context below. If not in context, say "I don't know".

Context:
{context}

Question:
{query}""", "stream": False}
    )

    return {
        "query": query,
        "retrieved": len(retrieved),
        "unique_hits": len(unique_hits),
        "context_chars": len(context),
        "top_scores": [round(h["rerank_score"], 3) for h in unique_hits[:3]],
        "answer": response.json().get("response", "error")[:300]
    }

print("=" * 60)
print("RAG PIPELINE EVALUATION")
print("=" * 60)

for i, query in enumerate(test_queries, 1):
    print(f"\nQuery {i}: {query}")
    print("-" * 40)
    
    try:
        result = run_query(query)
        print(f"Retrieved: {result['retrieved']} | Unique: {result['unique_hits']} | Chars: {result['context_chars']}")
        print(f"Top scores: {result['top_scores']}")
        print(f"Answer: {result['answer']}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n[ ] Retrieval: Good  OK  Bad")
    print("[ ] Answer:   Correct  Partial  Wrong")
    print("-" * 40)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)