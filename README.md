# TraceRAG

A traceable, citation-first document Q&A system with sentence-level precision.

TraceRAG retrieves answers from documents with exact, verbatim citations and structured source attribution. It is designed for high-trust use cases where accuracy, traceability, and auditability matter.

## Overview

TraceRAG is a Retrieval-Augmented Generation (RAG) pipeline that prioritises:

- precise sentence-level answers
- exact verbatim extraction
- structured source tracking (page, section, subsection)
- minimal, relevant context
- deterministic and explainable outputs

It avoids paraphrasing and focuses on returning what the source actually says.

## Features

- Sentence-level retrieval for high precision
- Source traceability (page, section, subsection)
- Verbatim citations (no summarisation or invention)
- Context sentences for explanation
- Noise filtering (figures, captions, equations)
- Query-aware ranking (what, why, how)
- Adaptive term and pattern boosting

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

## Prerequisites

- Python 3.10+
- [Milvus](https://milvus.io/docs/install_standalone-docker.md) running on `localhost:19530`
- [Ollama](https://ollama.com/) running with embedding model (e.g., `mxbai-embed-large`)

## Quick start

```bash
# Install
pip install -e .

# Ingest a PDF
tracerag ingest document.pdf

# Query the document
tracerag query "What is attention?"
```

Or using Python directly:

```python
from tracerag import TraceRAG

rag = TraceRAG()
rag.ingest_pdf("document.pdf")
results = rag.search("What is attention?")
```

## Example

Command:

```bash
python exercise/query.py "What is attention?"
```

Output:

```
============================================================
Answer
============================================================
Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

------------------------------------------------------------
Source
------------------------------------------------------------
Page: 2
Section: 2 Background

------------------------------------------------------------
Verbatim
------------------------------------------------------------
"Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence."

------------------------------------------------------------
Context
------------------------------------------------------------
"An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors."
```

## How it works

### Document ingestion

- Extract text from PDF
- Detect section and subsection headings
- Store metadata (page, section, subsection)

### Chunking

- Split into structured chunks
- Preserve metadata

### Retrieval

- Embed chunks using Ollama
- Store and search using Milvus

### Sentence ranking

- Split retrieved chunks into sentences
- Rank sentences against the query
- Apply query-type aware scoring

### Output

- Return best sentence as answer
- Include verbatim quote
- Attach source metadata
- Provide supporting context

## Design principles

- No hallucination: only return what exists in the source
- Traceability first: every answer must be attributable
- Precision over verbosity
- Minimal but sufficient context
- Deterministic behaviour where possible

## Use cases

- Technical document search
- Research and study
- Interview preparation
- High-trust Q&A systems
- Foundation for legal and compliance tools

## Current scope

TraceRAG is a high-precision retrieval system for structured documents.

It currently supports:

- single-document querying
- well-structured PDFs
- deterministic citation output

### Limitations

- No multi-document reasoning
- No conflict detection between sources
- No confidence scoring yet
- No "no-answer" safeguard for weak matches
- Assumes reasonably structured input documents

## Roadmap

- Confidence scoring for answers
- No-answer handling
- Multi-document support
- Conflict detection across sources
- Improved sentence validation
- Exportable audit reports

## Tech stack

- Python 3.10+
- [Milvus](https://milvus.io/) (vector database)
- [Ollama](https://ollama.com/) (embeddings)
- Custom sentence-level ranking

## Dependencies

- pymilvus>=2.4.0
- requests>=2.31.0
- numpy>=1.24.0
- pypdf (PDF extraction)

## Philosophy

TraceRAG is built on a simple premise:

> Answers should be grounded, attributable, and verifiable.
> Not generated. Retrieved.

## Licence

MIT (recommended for open-core adoption)

## Future direction

TraceRAG can serve as the foundation for:

- compliance-grade document systems
- legal research tools
- audit-ready AI assistants

The open-source core focuses on retrieval and traceability.
Advanced evaluation, audit, and enterprise features can be layered on top.