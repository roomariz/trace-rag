"""TraceRAG core package."""

import hashlib
import os
import re
from typing import Optional

import requests
from pypdf import PdfReader
from pymilvus import MilvusClient

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
MILVUS_URI = os.getenv("MILVUS_URI", "tcp://localhost:19530")


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        res = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": [text]},
            timeout=30,
        )
        res.raise_for_status()
        data = res.json()
        vectors.append(data["embeddings"][0])
    return vectors


def extract_sections(reader: PdfReader) -> list[dict]:
    sections = []
    current = {"heading": "Introduction", "text": "", "page": 0}

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r"^\d+\s+\w", line) and len(line) < 80:
                if current["text"]:
                    sections.append(current)
                current = {"heading": line, "text": "", "page": page_num}
            else:
                current["text"] += " " + line
        current["page"] = page_num

    if current["text"]:
        sections.append(current)
    return sections


def chunk_text(text: str, chunk_size: int = 500) -> list[dict]:
    section_pattern = r"^\d+\s+[A-Z]"
    subsection_pattern = r"^\d+\.\d+\s+[A-Z]"

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
                        "section": current_section,
                        "subsection": current_subsection,
                        "paragraph_id": paragraph_id,
                    })
                    paragraph_id += 1
                current_chunk = line

    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "section": current_section,
            "subsection": current_subsection,
            "paragraph_id": paragraph_id,
        })
    return chunks


class TraceRAG:
    def __init__(
        self,
        collection_name: str = "pdf_collection",
        milvus_uri: Optional[str] = None,
        ollama_url: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.client = MilvusClient(uri=milvus_uri or MILVUS_URI)
        self.ollama_url = ollama_url or OLLAMA_BASE_URL

    def ingest_pdf(self, pdf_path: str) -> int:
        reader = PdfReader(pdf_path)
        chunk_size = 500

        chunks = []
        current_section = ""
        current_subsection = ""
        current_chunk = ""
        paragraph_id = 0

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if not page_text:
                continue
            lines = page_text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                is_section = re.match(r"^\d+\s+\w", line) and len(line) < 80
                is_subsection = re.match(r"^\d+\.\d+\s+\w", line) and len(line) < 80

                if is_section:
                    current_section = line
                    current_subsection = ""
                elif is_subsection:
                    current_subsection = line
                else:
                    if len(current_chunk) + len(line) < chunk_size:
                        current_chunk += " " + line
                    else:
                        if current_chunk:
                            chunks.append({
                                "text": current_chunk.strip(),
                                "page": page_num + 1,
                                "section": current_section,
                                "subsection": current_subsection,
                                "paragraph_id": paragraph_id,
                            })
                            paragraph_id += 1
                        current_chunk = line

        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "page": page_num + 1,
                "section": current_section,
                "subsection": current_subsection,
                "paragraph_id": paragraph_id,
            })

        if not chunks:
            return 0

        dim = len(embed([chunks[0]["text"]])[0])

        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dim,
            metric_type="COSINE",
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
                "chunk_hash": compute_hash(c["text"]),
            }
            for i, c in enumerate(chunks)
        ]

        self.client.insert(collection_name=self.collection_name, data=data)
        self.client.flush(self.collection_name)
        self.client.load_collection(self.collection_name)

        return len(chunks)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        query_vector = embed([query])
        results = self.client.search(
            collection_name=self.collection_name,
            data=query_vector,
            limit=limit,
            output_fields=["text", "page", "section", "subsection"],
        )
        return [{"text": r["entity"]["text"], "score": r["distance"], **r["entity"]} for r in results[0]]