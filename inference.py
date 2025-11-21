import argparse
import json
import torch
import os
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# Import your prompt strategy
from prompts import create_qa_prompt

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

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
        print(f"Device: CUDA detected. Using 4-bit Unsloth model.")
        # Construct Unsloth model string
        model_name = f"unsloth/gemma-3-{size}-it-unsloth-bnb-4bit"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        return model_name, bnb_config, torch.bfloat16

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


def main(
    query_path: Path,
    paper_text_dir: Path,
    model_size: str,
    output_path: Path,
    batch_size: int = 4,  
) -> None:
    
    # Initialize Reader with the specific size
    reader = Reader(size=model_size)

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
        
        for i, item in enumerate(batch_items):
            paper_id = item.get("paper_id")
            question = item.get("question")
            
            txt_file_path = paper_text_dir / f"{paper_id}.txt"
            
            if txt_file_path.exists():
                try:
                    paper_content = txt_file_path.read_text(encoding="utf-8")
                    
                    # --- USE PROMPT FROM prompts.py ---
                    prompt = create_qa_prompt(paper_content, question)
                    
                    batch_prompts.append(prompt)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Error reading file {txt_file_path}: {e}")
            else:
                print(f"Warning: File not found {txt_file_path}")

        if batch_prompts:
            batch_answers = reader.batch_generate(batch_prompts)
            
            for local_idx, answer in zip(valid_indices, batch_answers):
                original_item = batch_items[local_idx]
                outputs.append({
                    "paper_id": original_item.get("paper_id"),
                    "question": original_item.get("question"),
                    "citation_index": original_item.get("citation_index", None),
                    "prediction": answer
                })
        
        print(f"Processed {min(idx + batch_size, len(queries_data))}/{len(queries_data)}", flush=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    
    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--query_path", type=Path, required=True)
    parser.add_argument("--paper_text_dir", type=Path, required=True)
    
    # CHANGED: Instead of full model name, we just take the size
    parser.add_argument("--model_size", type=str, default="4b", 
                        choices=MODEL_SIZES,
                        help="Model size: 270m, 1b, 4b, or 27b")
    
    parser.add_argument("--output_path", type=Path, default=Path("predictions.json"))
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    main(**vars(args))