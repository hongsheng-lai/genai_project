# GenAI Project


This project automatically optimizes model loading based on your hardware:

  * **NVIDIA GPUs (CUDA):** Loads 4-bit quantized models (via `unsloth`) for maximum efficiency.
  * **Apple Silicon (Mac/MPS):** Loads standard models in `bfloat16` (via `google`).

## 📂 Project Structure

```text
.
├── QA/                 # Input .jsonl files
├── papers_text/        # Text files named by paper ID
├── predictions/        # Output JSON files (generated automatically)
├── inference.py    # Main script
├── prompts.py          # Prompt engineering logic
└── run_citation_qa.sh       # Batch processing script
```


## 🚀 Usage

### 1\. Run All QA Files

Automatically processes all `.jsonl` files in the `QA/` folder.

```bash
# Default (uses 4b model)
./run_citation_qa.sh

# Specific size (options: 270m, 1b, 4b, 27b)
./run_citation_qa.sh 270m
```

### 2\. Run Single File (Manual)

Run a specific file with custom settings.

```bash
python inference.py \
  --query_path QA/my_questions.jsonl \
  --paper_text_dir papers_text \
  --model_size 4b \
  --batch_size 1
```

### Evaluation
Evaluate predictions against ground truth.

```bash
python evaluation.py --ground_truth <GROUND_TRUTH_JSON_FILE> --predictions <PREDICTIONS_FILE>
```