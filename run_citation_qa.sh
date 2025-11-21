#!/bin/bash

# --- Configuration ---
QA_DIR="QA"
TEXT_DIR="papers_text"
OUTPUT_DIR="predictions"
BATCH_SIZE=1

# You can pass the model size: 270m, 1b, 4b, 27b
# Usage: ./run_all_qa.sh 27b
MODEL_SIZE=${1:-"270m"} 

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Starting Batch Inference"
echo "Model Size: $MODEL_SIZE"
echo "QA Directory: $QA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "============================================"

# Loop through all .jsonl files in the QA directory
for query_file in "$QA_DIR"/*.jsonl; do
    
    # Check if file exists (in case directory is empty)
    [ -e "$query_file" ] || continue

    # Extract filename without extension for the output name
    # e.g., "QA/data_1.jsonl" -> "data_1"
    filename=$(basename -- "$query_file")
    filename_no_ext="${filename%.*}"
    
    # Define output path
    output_file="$OUTPUT_DIR/${filename_no_ext}_pred.json"

    echo ""
    echo ">>> Processing: $filename"
    echo ">>> Saving to: $output_file"

    # Run the Python script
    python inference.py \
        --query_path "$query_file" \
        --paper_text_dir "$TEXT_DIR" \
        --model_size "$MODEL_SIZE" \
        --output_path "$output_file" \
        --batch_size "$BATCH_SIZE"

    # Optional: Check if python script failed
    if [ $? -ne 0 ]; then
        echo "!!! Error processing $filename"
    else
        echo ">>> Done with $filename"
    fi

done

echo ""
echo "============================================"
echo "All tasks completed."
echo "============================================"