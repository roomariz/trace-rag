#!/usr/bin/env python
"""Interactive RAG query CLI"""

import sys
import re
import requests
import hashlib
from pypdf import PdfReader
from pymilvus import MilvusClient

def compute_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

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

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    result = []
    for s in sentences:
        s = s.strip()
        s_lower = s.lower()
        
        if len(s) < 35:
            continue
        if s[-1] not in '.!?':
            continue
        if s_lower.startswith('figure') or s_lower.startswith('table'):
            continue
        if s_lower.startswith('('):
            continue
        if re.match(r'^\d+[\s.]', s):
            continue
        if '[{' in s or ']}' in s:
            continue
        if 'equation' in s_lower or 'formula' in s_lower:
            continue
        
        words = s.split()
        if len([w for w in words if w.istitle()]) > len(words) * 0.5:
            continue
            
        result.append(s)
    return result

def rank_sentences_in_chunks(reranked, query_vector, query, limit_chars=2000):
    seen = set()
    ranked_sentences = []
    query_lower = query.lower()
    query_terms = set(query_lower.split())
    
    is_why = query_lower.startswith("why")
    is_how = query_lower.startswith("how")
    is_what = query_lower.startswith("what")
    
    for chunk in reranked:
        sentences = split_into_sentences(chunk["text"])
        if not sentences:
            continue
        
        section = chunk.get("section", "")
        subsection = chunk.get("subsection", "")
        
        section_depth = 0
        if subsection and "." in subsection:
            section_depth = len(subsection.split(".")) - 1
        elif section and "." in section:
            section_depth = len(section.split(".")) - 1
        
        sent_vectors = embed(sentences)
        
        for i, sent in enumerate(sentences):
            if sent in seen:
                continue
            
            sim = sum(a * b for a, b in zip(query_vector[0], sent_vectors[i]))
            
            sent_lower = sent.lower()
            term_match = sum(1 for term in query_terms if term in sent_lower and len(term) > 4)
            sim += min(term_match * 0.1, 0.25)
            
            if "allows" in sent_lower:
                sim += 0.35
            
            if is_what:
                if "is an" in sent_lower or "is a" in sent_lower:
                    sim += 0.2
            
            if is_why or is_how:
                if "used to" in sent_lower:
                    sim += 0.055
                if "enables" in sent_lower:
                    sim += 0.1
                if "used to" in sent_lower:
                    sim += 0.05
                
                if "constant number" in sent_lower:
                    sim -= 0.2
                if "averaging" in sent_lower:
                    sim -= 0.15
            
            ranked_sentences.append({
                "text": sent,
                "score": sim,
                "page": chunk.get("page", 0),
                "section": chunk.get("section", ""),
                "subsection": chunk.get("subsection", ""),
                "chunk_id": chunk.get("id", 0),
                "sentence_idx": i
            })
            seen.add(sent)
    
    ranked_sentences.sort(key=lambda x: x["score"], reverse=True)
    return ranked_sentences


def answer_query(query):
    pdf_path = "sample.pdf"
    current_hash = compute_file_hash(pdf_path)

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
        print("PDF changed. Run pdf_rag_milvus.py first to re-ingest.")
        return

    SYNTHESIS_KEYWORDS = ["why", "problem", "compare", "advantage", "difference", "how does", "benefits"]
    is_synthesis = any(kw in query.lower() for kw in SYNTHESIS_KEYWORDS)

    INITIAL_LIMIT = 15 if is_synthesis else 10
    MAX_CONTEXT_CHARS = 2500 if is_synthesis else 2000

    query_vector = embed([query])

    results = client.search(
        collection_name="pdf_collection",
        data=query_vector,
        limit=INITIAL_LIMIT,
        output_fields=["id", "text", "page", "section", "subsection"],
    )

    candidates = []
    for h in results[0]:
        entity = h["entity"]
        candidates.append({
            "text": entity.get("text", ""),
            "score": h["distance"],
            "id": h["id"],
            "page": entity.get("page", 0),
            "section": entity.get("section", ""),
            "subsection": entity.get("subsection", "")
        })

    candidate_texts = [c["text"] for c in candidates]
    candidate_vectors = embed(candidate_texts)

    reranked = []
    for i, c in enumerate(candidates):
        sim = sum(a * b for a, b in zip(query_vector[0], candidate_vectors[i]))
        c["rerank_score"] = sim
        reranked.append(c)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    sentence_results = rank_sentences_in_chunks(reranked, query_vector, query, MAX_CONTEXT_CHARS)

    if not sentence_results:
        print("No results found.")
        return

    answer_sent = sentence_results[0]
    best_answer = answer_sent["text"]
    
    context_sent = None
    for s in sentence_results[1:4]:
        if s["score"] > 0.5 and s["text"] != answer_sent["text"]:
            context_sent = s
            break
    
    print("\n" + "=" * 60)
    print("Answer")
    print("=" * 60)
    print(best_answer)
    
    print("\n" + "-" * 60)
    print("Source")
    print("-" * 60)
    print(f"Page: {answer_sent.get('page', 'N/A')}")
    print(f"Section: {answer_sent.get('section', 'N/A')}")
    if answer_sent.get("subsection"):
        print(f"Subsection: {answer_sent.get('subsection', 'N/A')}")
    
    print("\n" + "-" * 60)
    print("Verbatim")
    print("-" * 60)
    print(f'"{answer_sent["text"]}"')
    
    if context_sent:
        print("\n" + "-" * 60)
        print("Context")  
        print("-" * 60)
        print(f'"{context_sent["text"]}"')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Usage: python query.py \"your question here\"")
        print("Example: python query.py \"What is self-attention?\"")
        sys.exit(1)

    answer_query(query)