import subprocess
import sys
import time

test_queries = [
    "What is attention in transformers?",
    "What is self-attention?",
    "Why is multi-head attention used?",
    "What problem does the Transformer solve?",
    "How does encoder-decoder attention work?",
    "What is the difference between encoder and decoder?",
    "Why is the Transformer faster than RNNs?",
]

print("=" * 60)
print("RAG PIPELINE EVALUATION")
print("=" * 60)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'=' * 60}")
    print(f"Query {i}: {query}")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "pdf_rag_milvus.py"],
        capture_output=True,
        text=True,
        input=query + "\n",
        timeout=120
    )
    
    output = result.stdout
    
    in_answer = False
    answer_lines = []
    for line in output.split("\n"):
        if "--- Answer ---" in line:
            in_answer = True
            continue
        if in_answer and line.strip():
            answer_lines.append(line)
    
    answer = " ".join(answer_lines[:3]) if answer_lines else "(no answer)"
    
    in_results = False
    result_lines = []
    for line in output.split("\n"):
        if "--- Result ---" in line:
            in_results = True
            continue
        if in_results and line.strip() and "Score:" in line:
            result_lines.append(line)
        if in_results and "--- Answer ---" in line:
            break
    
    print(f"Retrieved chunks: {len(result_lines)}")
    print(f"Answer: {answer[:200]}")
    
    print("\n[Evaluate]")
    print("Retrieval:  [ ] Good  [ ] OK  [ ] Bad")
    print("Answer:   [ ] Correct  [ ] Partial  [ ] Wrong")
    
    time.sleep(0.5)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)