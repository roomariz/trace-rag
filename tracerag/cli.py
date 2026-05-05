"""TraceRAG CLI."""

import argparse
import sys

from tracerag import TraceRAG


def main():
    parser = argparse.ArgumentParser(description="TraceRAG - Traceable document Q&A")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest a PDF document")
    ingest.add_argument("pdf", help="Path to PDF file")
    ingest.add_argument("--collection", default="pdf_collection", help="Collection name")

    query = subparsers.add_parser("query", help="Query the document")
    query.add_argument("question", help="Question to ask")
    query.add_argument("--collection", default="pdf_collection", help="Collection name")
    query.add_argument("--limit", type=int, default=10, help="Number of results")

    args = parser.parse_args()

    rag = TraceRAG(collection_name=args.collection)

    if args.command == "ingest":
        count = rag.ingest_pdf(args.pdf)
        print(f"Ingested {count} chunks")

    elif args.command == "query":
        results = rag.search(args.question, limit=args.limit)
        for r in results:
            print(f"\n--- Result (score: {r['score']:.3f}) ---")
            print(f"Page: {r.get('page', 'N/A')}")
            print(f"Section: {r.get('section', 'N/A')}")
            print(r["text"][:300])


if __name__ == "__main__":
    main()