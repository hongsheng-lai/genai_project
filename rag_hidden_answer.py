import argparse
import json
import re
import sys
import os

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

import torch
import numpy as np
from tqdm import tqdm

# NLP & Retrieval
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    import bm25s
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Missing dependencies. Please install: pip install sentence-transformers bm25s faiss-cpu nltk")
    sys.exit(1)

# Transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        pass # punkt_tab might not be available in older nltk

# -----------------------------------------------------------------------------
# 1. Data Processing (from anlp-hw2/rag/data.py)
# -----------------------------------------------------------------------------

class WebData:
    def __init__(self, data_path: Path, chunk_size: int = 512, overlap: int = 128):
        self.data_path = data_path
        if self.data_path.is_file():
            self.files = [self.data_path]
        else:
            self.files = list(self.data_path.rglob("*.txt")) + list(self.data_path.rglob("*.json"))
        self.documents = []
        self.chunks = []
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self._load_documents()
        self._chunk_documents()
    
    def _load_documents(self):
        print(f"Loading documents from {self.data_path}...")
        for file in tqdm(self.files, desc="Loading files"):
            if file.suffix == ".txt":
                with file.open(mode="r", encoding="utf-8") as f:
                    self.documents.append(f.read())
            elif file.suffix == ".json":
                with file.open(mode="r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)                     
                        if isinstance(data, list):
                            for item in data:
                                text = self.json_to_text(item)
                                if text:
                                    self.documents.append(text)
                        else:
                            text = self.json_to_text(data)
                            if text:
                                self.documents.append(text)
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON file: {file}")

    def json_to_text(self, data, prefix="") -> str:
        if isinstance(data, dict):
            text_parts = []
            for key, value in data.items():
                formatted_key = key.replace("_", " ").title()  
                if isinstance(value, dict):                
                    nested_text = self.json_to_text(value, prefix=formatted_key)
                    if nested_text:
                        text_parts.append(nested_text)
                elif isinstance(value, list):
                    for item in value:
                        nested_text = self.json_to_text(item)
                        if nested_text:
                            text_parts.append(nested_text)
                elif value:
                    text_parts.append(f"{formatted_key}: {value}")
            return ". ".join(text_parts) + "." if text_parts else ""
        elif isinstance(data, str):
            return data
        elif isinstance(data, (int, float, bool)):
            return str(data)
        else:
            return ""

    def _chunk_documents(self):
        print("Chunking documents...")
        for doc in tqdm(self.documents, desc="Chunking"):
            sentences = sent_tokenize(doc)
            chunks = []
            curr_idx = 0
            curr_length = 0
            current_chunk_sents = []

            while curr_idx < len(sentences):
                sent = sentences[curr_idx]
                
                # Simple length estimation by chars, or could use tokens. 
                # The original code used len(sent) which is chars.
                if curr_length + len(sent) > self.chunk_size and current_chunk_sents:
                    chunk_text = " ".join(current_chunk_sents).strip()
                    self.chunks.append(chunk_text)

                    # Handle overlap
                    while curr_length > self.overlap and len(current_chunk_sents) > 1:
                        rm_sent = current_chunk_sents.pop(0)
                        curr_length -= len(rm_sent)
                
                current_chunk_sents.append(sent)
                curr_length += len(sent)
                curr_idx += 1

            if current_chunk_sents:
                chunk_text = " ".join(current_chunk_sents).strip()
                self.chunks.append(chunk_text)
    
    def get_chunks(self):
        return self.chunks

# -----------------------------------------------------------------------------
# 2. Model Components (Embedder, Retriever) - Adapted for MPS
# -----------------------------------------------------------------------------

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"):
        self.device = "cpu"
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                # MPS is often unstable for embedding generation in parallel with LLM
                # Defaulting to CPU for Embedder is safer and usually fast enough
                print("⚠️ MPS detected but defaulting Embedder to CPU for stability.")
                self.device = "cpu" 
        else:
            self.device = device
        
        print(f"✅ Embedder using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def __call__(self, chunks: List[str]) -> np.ndarray:
        # MPS batch size handling
        batch_size = 64 if self.device == "cuda" else 32
        
        embs = self.model.encode(
            chunks, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            batch_size=batch_size,
            device=self.device
        )
        return embs

class Retriever:
    def __init__(
        self,
        documents: List[str],
        embedder: Optional[Embedder] = None,
        retrieval_type: str = "hybrid",
        top_k: int = 5,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        self.documents = documents
        self.embedder = embedder
        self.retrieval_type = retrieval_type
        self.top_k = top_k
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

        self.bm25_index = None
        if retrieval_type in ["sparse", "hybrid"]:
            self._build_bm25_index()

        self.faiss_index = None
        self.doc_embs = None
        if retrieval_type in ["dense", "hybrid"]:
            if embedder is None:
                raise ValueError("Embedder is required for dense or hybrid retrieval")
            self._build_faiss_index()

    def _build_bm25_index(self) -> None:
        print("Building BM25 index...")
        corpus_tokens = bm25s.tokenize(self.documents, stopwords="en", show_progress=True)
        self.bm25_index = bm25s.BM25()
        self.bm25_index.index(corpus_tokens, show_progress=True)

    def _build_faiss_index(self) -> None:
        print("Building FAISS index...")
        self.doc_embs = self.embedder(self.documents)
        emb_dim = self.doc_embs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(emb_dim)
        faiss.normalize_L2(self.doc_embs)
        self.faiss_index.add(self.doc_embs)

    def _sparse_retrieve(self, query: str) -> Tuple[List[int], List[float]]:
        query_tokens = bm25s.tokenize(query, stopwords="en")
        doc_ids, scores = self.bm25_index.retrieve(query_tokens, k=self.top_k)
        return doc_ids[0].tolist(), scores[0].tolist()

    def _dense_retrieve(self, query: str) -> Tuple[List[int], List[float]]:
        query_emb = self.embedder([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.faiss_index.search(query_emb, self.top_k)
        return indices[0].tolist(), scores[0].tolist()

    def _hybrid_retrieve(self, query: str) -> List[str]:
        # Get more candidates for hybrid reranking
        k_factor = 2
        temp_k = self.top_k * k_factor
        
        # Sparse
        query_tokens = bm25s.tokenize(query, stopwords="en")
        sparse_ids, sparse_scores = self.bm25_index.retrieve(query_tokens, k=temp_k)
        sparse_ids = sparse_ids[0]
        sparse_scores = sparse_scores[0]
        
        # Dense
        query_emb = self.embedder([query])
        faiss.normalize_L2(query_emb)
        dense_scores, dense_ids = self.faiss_index.search(query_emb, temp_k)
        dense_ids = dense_ids[0]
        dense_scores = dense_scores[0]

        # Normalize scores
        def normalize(scores):
            if len(scores) == 0: return scores
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s: return [1.0] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        sparse_scores_norm = normalize(sparse_scores)
        dense_scores_norm = normalize(dense_scores)

        # Combine
        combined_scores = {}
        
        for idx, score in zip(sparse_ids, sparse_scores_norm):
            combined_scores[idx] = combined_scores.get(idx, 0) + score * self.sparse_weight
            
        for idx, score in zip(dense_ids, dense_scores_norm):
            combined_scores[idx] = combined_scores.get(idx, 0) + score * self.dense_weight
            
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [idx for idx, _ in sorted_ids[:self.top_k]]
        
        return [self.documents[i] for i in top_ids]

    def __call__(self, query: str) -> List[str]:
        if self.retrieval_type == "sparse":
            ids, _ = self._sparse_retrieve(query)
            return [self.documents[i] for i in ids]
        elif self.retrieval_type == "dense":
            ids, _ = self._dense_retrieve(query)
            return [self.documents[i] for i in ids]
        else:
            return self._hybrid_retrieve(query)

# -----------------------------------------------------------------------------
# 3. Reader (LLM) - Adapted from inference.py, removing 4-bit
# -----------------------------------------------------------------------------

MODEL_SIZES = ["270m", "1b", "4b", "27b"]

def get_model_config(size: str, device: str = "auto"):
    if size not in MODEL_SIZES:
        raise ValueError(f"Invalid size. Choose from: {MODEL_SIZES}")

    if device == "cpu":
        print(f"Device: CPU forced.")
        model_name = f"google/gemma-3-{size}-it"
        return model_name, torch.float32

    if torch.cuda.is_available():
        print(f"Device: CUDA detected. Using standard model (no 4-bit requested).")
        model_name = f"google/gemma-3-{size}-it"
        return model_name, torch.bfloat16

    elif torch.backends.mps.is_available():
        print(f"Device: MPS (Mac) detected. Using Google standard model.")
        model_name = f"google/gemma-3-{size}-it"
        return model_name, torch.bfloat16

    else:
        print(f"Device: CPU detected.")
        model_name = f"google/gemma-3-{size}-it"
        return model_name, torch.float32

class Reader:
    def __init__(self, size: str = "4b", device: str = "auto") -> None:
        self.model_name, self.dtype = get_model_config(size, device)
        print(f"Loading Reader model: {self.model_name}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            low_cpu_mem_usage=True,
            device_map=device if device != "auto" else "auto", 
            torch_dtype=self.dtype
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256, 
            batch_size=1,       
        )

    def batch_generate(self, prompts: List[str]) -> List[str]:
        outputs = self.llm_pipe(
            prompts,
            batch_size=min(len(prompts), 8),
            return_full_text=False,
            do_sample=False,
            temperature=None,
            top_p=None
        )
        answers = []
        for output in outputs:
            if isinstance(output, list):
                answers.append(output[0]["generated_text"].strip())
            else:
                answers.append(output["generated_text"].strip())
        return answers

# -----------------------------------------------------------------------------
# 4. Hidden Answer Parsing (from hidden_answer_inference.py)
# -----------------------------------------------------------------------------

def parse_hidden_answer_file(file_path: Path) -> Tuple[List[Tuple[int, str]], Dict[int, str]]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    problems = []
    answers = {}
    in_answers = False
    problem_index = 0
    answer_pattern = re.compile(r"^\[(\d+)\]\s*Answer:\s*(.+?)\s*$")

    for raw_line in lines:
        line = raw_line.strip()
        if not line: continue
        if line.startswith("=== QUESTIONS ==="): continue
        if line.startswith("=== ANSWERS ==="):
            in_answers = True
            continue

        if not in_answers:
            problem_index += 1
            problems.append((problem_index, line))
        else:
            m = answer_pattern.match(line)
            if m:
                idx = int(m.group(1))
                val = m.group(2).strip()
                answers[idx] = val
    return problems, answers

_NUM_REGEX = re.compile(r"""^[\s\S]*?(?P<num>[+-]?(?:(?:\d+\.\d*|\.\d+|\d+))(?:[eE][+-]?\d+)?)""", re.VERBOSE)

def extract_first_number(text: str) -> Optional[str]:
    m = _NUM_REGEX.match(text.strip())
    if not m: return None
    num = m.group("num")
    if num.endswith("."): num = num[:-1]
    if num == "+0": num = "0"
    return num

def to_number(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", ""))
    except:
        return None

def is_integer_like(value: str) -> bool:
    try:
        f = float(value)
        return float(int(f)) == f
    except:
        return False

def compare_numbers(pred: str, gt: str, atol: float = 1e-6, rtol: float = 1e-4) -> Tuple[bool, Optional[float]]:
    if is_integer_like(pred) and is_integer_like(gt):
        return int(float(pred)) == int(float(gt)), abs(int(float(pred)) - int(float(gt)))
    p = to_number(pred)
    g = to_number(gt)
    if p is None or g is None: return False, None
    abs_err = abs(p - g)
    if abs_err <= max(atol, rtol * abs(g)): return True, abs_err
    return False, abs_err

# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------

def build_rag_prompt(context: str, question: str) -> str:
    return (
        "You are a precise calculator. Use the provided context if helpful, but primarily solve the math problem. "
        "Return ONLY the final numeric answer without any words or units.\n\n"
        f"Context:\n{context}\n\n"
        f"Problem: {question}\n"
        "Answer:"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, default=Path("hidden_answer.txt"))
    parser.add_argument("--corpus_path", type=Path, default=Path("hidden_answer.txt"))
    parser.add_argument("--output_path", type=Path, default=Path("rag_hidden_answer_predictions.json"))
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to process")
    parser.add_argument("--model_size", type=str, default="4b", choices=MODEL_SIZES)
    parser.add_argument("--embedder_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--retrieval_type", type=str, default="dense", choices=["sparse", "dense", "hybrid"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--sparse_weight", type=float, default=0.5)
    parser.add_argument("--dense_weight", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Force device usage")
    args = parser.parse_args()

    # Load Corpus
    if not args.corpus_path.exists():
        print(f"Error: Corpus path {args.corpus_path} does not exist.")
        return
    web_data = WebData(args.corpus_path, chunk_size=args.chunk_size, overlap=args.overlap)
    documents = web_data.get_chunks()
    print(f"Loaded {len(documents)} chunks.")

    # Init Embedder/Retriever
    embedder = None
    if args.retrieval_type in ["dense", "hybrid"]:
        embedder = Embedder(model_name=args.embedder_model, device=args.device)
    
    retriever = Retriever(
        documents=documents,
        embedder=embedder,
        retrieval_type=args.retrieval_type,
        top_k=args.top_k,
        sparse_weight=args.sparse_weight,
        dense_weight=args.dense_weight
    )

    # Init Reader
    reader = Reader(size=args.model_size, device=args.device)

    # Load Questions
    problems, answers = parse_hidden_answer_file(args.input_path)
    has_answers = len(answers) > 0
    print(f"Loaded {len(problems)} questions.")

    # Run
    results = []
    num_correct = 0
    
    all_problems = problems
    if args.limit:
        all_problems = problems[:args.limit]
        print(f"Limiting to {args.limit} samples.")

    num_batches = (len(all_problems) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(all_problems))
        batch_problems = all_problems[start_idx:end_idx]
        
        batch_prompts = []
        batch_indices = []
        
        for idx, question_text in batch_problems:
            retrieved_chunks = retriever(question_text)
            context = "\n\n".join(retrieved_chunks)
            prompt = build_rag_prompt(context, question_text)
            batch_prompts.append(prompt)
            batch_indices.append((idx, question_text, retrieved_chunks))
        
        batch_outputs = reader.batch_generate(batch_prompts)
        
        for (idx, question_text, retrieved_chunks), raw_output in zip(batch_indices, batch_outputs):
            parsed = extract_first_number(raw_output)
            gt_str = answers.get(idx) if has_answers else None
            correct = None
            abs_error = None
            if has_answers and parsed is not None and gt_str is not None:
                correct, abs_error = compare_numbers(parsed, gt_str)
                if correct: num_correct += 1
            
            results.append({
                "index": idx,
                "question": question_text,
                "prediction_raw": raw_output,
                "prediction_parsed": parsed,
                "answer_gt": gt_str,
                "correct": correct,
                "abs_error": abs_error,
                "retrieved_context": retrieved_chunks
            })

    summary = {
        "total": len(all_problems),
        "evaluated": len([p for p in all_problems if p[0] in answers]) if has_answers else 0,
        "num_correct": num_correct if has_answers else None,
        "accuracy": (num_correct / len(all_problems)) if has_answers and len(all_problems) > 0 else None,
    }
    
    output_data = {"summary": summary, "results": results}
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"\nResults saved to {args.output_path}")
    if summary["accuracy"] is not None:
        print(f"Accuracy: {summary['accuracy']:.4f} ({summary['num_correct']}/{summary['total']})")

if __name__ == "__main__":
    main()
