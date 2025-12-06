import argparse
import json
import torch
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm

# Import your prompt strategy
from prompts import create_qa_prompt

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

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
        pass 

# --- Configuration ---
MODEL_SIZES = ["270m", "1b", "4b", "27b"]

def get_model_config(size: str):
    """
    Returns (model_name, quantization_config, torch_dtype) based on device and size.
    """
    if size not in MODEL_SIZES:
        raise ValueError(f"Invalid size. Choose from: {MODEL_SIZES}")

    # 1. Check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        print(f"Device: CUDA detected. Using Google standard model.")
        # Construct Google model string
        model_name = f"google/gemma-3-{size}-it"
        
        # MPS does not support bitsandbytes 4-bit yet. 
        # We return None for config, but force bfloat16 to prevent NaNs.
        return model_name, None, torch.bfloat16

    # 2. Check for MPS (Mac Silicon)
    elif torch.backends.mps.is_available():
        print(f"Device: MPS (Mac) detected. Using Google standard model.")
        print("Note: Using bfloat16 for MPS stability.")
        # Construct Google model string
        model_name = f"google/gemma-3-{size}-it"
        
        # MPS does not support bitsandbytes 4-bit yet. 
        # We return None for config, but force bfloat16 to prevent NaNs.
        return model_name, None, torch.bfloat16

    # 3. CPU Fallback
    else:
        print(f"Device: CPU detected.")
        model_name = f"google/gemma-3-{size}-it"
        return model_name, None, torch.float32


class Reader:
    def __init__(self, size: str = "4b") -> None:
        
        # Get the specific configuration for this hardware/size combination
        self.model_name, self.bnb_config, self.dtype = get_model_config(size)
        
        print(f"Loading model: {self.model_name}...")
        
        # Prepare kwargs
        model_kwargs = {"torch_dtype": self.dtype}
        if self.bnb_config:
            model_kwargs["quantization_config"] = self.bnb_config

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            low_cpu_mem_usage=True,
            device_map="auto", 
            **model_kwargs
        )

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Setup Pipeline
        self.llm_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256, 
            batch_size=1,       
        )

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Batch generate using greedy decoding (do_sample=False) to avoid errors.
        """
        outputs = self.llm_pipe(
            prompts,
            batch_size=min(len(prompts), 8),
            return_full_text=False,
            do_sample=False,    # Critical for stability
            temperature=None,
            top_p=None
        )
        
        # Clean output
        answers = []
        for output in outputs:
            # Pipeline sometimes returns list of list, sometimes list of dicts depending on version
            if isinstance(output, list):
                answers.append(output[0]["generated_text"].strip())
            else:
                answers.append(output["generated_text"].strip())
            
        return answers

# -----------------------------------------------------------------------------
# RAG Components
# -----------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    curr_idx = 0
    curr_length = 0
    current_chunk_sents = []

    while curr_idx < len(sentences):
        sent = sentences[curr_idx]
        
        if curr_length + len(sent) > chunk_size and current_chunk_sents:
            chunk_text = " ".join(current_chunk_sents).strip()
            chunks.append(chunk_text)

            # Handle overlap
            while curr_length > overlap and len(current_chunk_sents) > 1:
                rm_sent = current_chunk_sents.pop(0)
                curr_length -= len(rm_sent)
        
        current_chunk_sents.append(sent)
        curr_length += len(sent)
        curr_idx += 1

    if current_chunk_sents:
        chunk_text = " ".join(current_chunk_sents).strip()
        chunks.append(chunk_text)
    return chunks

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"):
        self.device = "cpu"
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                print("⚠️ MPS detected but defaulting Embedder to CPU for stability.")
                self.device = "cpu" 
        else:
            self.device = device
        
        print(f"✅ Embedder using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def __call__(self, chunks: List[str]) -> np.ndarray:
        batch_size = 64 if self.device == "cuda" else 32
        embs = self.model.encode(
            chunks, 
            show_progress_bar=False, 
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
        # print("Building BM25 index...")
        corpus_tokens = bm25s.tokenize(self.documents, stopwords="en", show_progress=False)
        self.bm25_index = bm25s.BM25()
        self.bm25_index.index(corpus_tokens, show_progress=False)

    def _build_faiss_index(self) -> None:
        # print("Building FAISS index...")
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
        k_factor = 2
        temp_k = self.top_k * k_factor
        
        query_tokens = bm25s.tokenize(query, stopwords="en")
        sparse_ids, sparse_scores = self.bm25_index.retrieve(query_tokens, k=temp_k)
        sparse_ids = sparse_ids[0]
        sparse_scores = sparse_scores[0]
        
        query_emb = self.embedder([query])
        faiss.normalize_L2(query_emb)
        dense_scores, dense_ids = self.faiss_index.search(query_emb, temp_k)
        dense_ids = dense_ids[0]
        dense_scores = dense_scores[0]

        def normalize(scores):
            if len(scores) == 0: return scores
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s: return [1.0] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        sparse_scores_norm = normalize(sparse_scores)
        dense_scores_norm = normalize(dense_scores)

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
# Main
# -----------------------------------------------------------------------------

def main(
    query_path: Path,
    paper_text_dir: Path,
    model_size: str,
    output_path: Path,
    batch_size: int = 4,
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    retrieval_type: str = "dense",
    top_k: int = 5,
    chunk_size: int = 512,
    overlap: int = 128,
    sparse_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> None:
    
    # Initialize Reader
    reader = Reader(size=model_size)

    # Initialize Embedder (if needed)
    embedder = None
    if retrieval_type in ["dense", "hybrid"]:
        embedder = Embedder(model_name=embedder_model)

    # Load Queries
    print(f"Loading queries from {query_path}...")
    queries_data = []
    with query_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries_data.append(json.loads(line))

    outputs = []
    print(f"Processing {len(queries_data)} questions...")
    
    for idx in range(0, len(queries_data), batch_size):
        batch_items = queries_data[idx:idx + batch_size]
        batch_prompts = []
        valid_indices = [] 
        batch_retrieved_contexts = []
        
        for i, item in enumerate(batch_items):
            paper_id = item.get("paper_id")
            question = item.get("question")
            
            txt_file_path = paper_text_dir / f"{paper_id}.txt"
            
            if txt_file_path.exists():
                try:
                    paper_content = txt_file_path.read_text(encoding="utf-8")
                    
                    # --- RAG LOGIC ---
                    # 1. Chunk
                    chunks = chunk_text(paper_content, chunk_size=chunk_size, overlap=overlap)
                    
                    # 2. Retrieve
                    if not chunks:
                        print(f"Warning: No chunks found for {paper_id}")
                        retrieved_chunks = []
                    else:
                        retriever = Retriever(
                            documents=chunks,
                            embedder=embedder,
                            retrieval_type=retrieval_type,
                            top_k=top_k,
                            sparse_weight=sparse_weight,
                            dense_weight=dense_weight
                        )
                        retrieved_chunks = retriever(question)
                    
                    batch_retrieved_contexts.append(retrieved_chunks)

                    # 3. Construct Prompt
                    context = "\n\n".join(retrieved_chunks)
                    prompt = create_qa_prompt(context, question)
                    
                    batch_prompts.append(prompt)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Error processing file {txt_file_path}: {e}")
            else:
                print(f"Warning: File not found {txt_file_path}")

        if batch_prompts:
            batch_answers = reader.batch_generate(batch_prompts)
            
            for local_idx, answer, retrieved_chunks in zip(valid_indices, batch_answers, batch_retrieved_contexts):
                original_item = batch_items[local_idx]
                outputs.append({
                    "paper_id": original_item.get("paper_id"),
                    "question": original_item.get("question"),
                    "citation_index": original_item.get("citation_index", None),
                    "prediction": answer,
                    "retrieved_context": retrieved_chunks
                })
        
        print(f"Processed {min(idx + batch_size, len(queries_data))}/{len(queries_data)}", flush=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    
    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--query_path", type=Path, required=True)
    parser.add_argument("--paper_text_dir", type=Path, required=True)
    parser.add_argument("--model_size", type=str, default="4b", choices=MODEL_SIZES)
    parser.add_argument("--output_path", type=Path, default=Path("predictions_rag.json"))
    parser.add_argument("--batch_size", type=int, default=1)
    
    # RAG Arguments
    parser.add_argument("--embedder_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--retrieval_type", type=str, default="hybrid", choices=["sparse", "dense", "hybrid"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--sparse_weight", type=float, default=0.5)
    parser.add_argument("--dense_weight", type=float, default=0.5)
    
    args = parser.parse_args()
    main(**vars(args))
