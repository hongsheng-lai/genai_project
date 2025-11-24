import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

def normalize_text(text):
    """
    Lowercases, removes punctuation, and standardizes whitespace.
    'L. Baulieu' -> 'l baulieu'
    'Ostrogradski, M.V.' -> 'ostrogradski mv'
    """
    if not text:
        return ""
    # Lowercase
    text = str(text).lower()
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_f1(truth, prediction):
    """
    Computes Token-level F1 score.
    Useful for partial matches like 'Baulieu' vs 'L. Baulieu'.
    """
    truth_tokens = normalize_text(truth).split()
    pred_tokens = normalize_text(prediction).split()

    if not truth_tokens or not pred_tokens:
        return 0.0

    common = Counter(truth_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_token_sort_ratio(truth, prediction):
    """
    Sorts tokens alphabetically to handle 'Last, First' vs 'First Last' scenarios.
    Returns a similarity ratio (0.0 to 1.0).
    """
    t_tokens = sorted(normalize_text(truth).split())
    p_tokens = sorted(normalize_text(prediction).split())
    
    t_str = " ".join(t_tokens)
    p_str = " ".join(p_tokens)
    
    return SequenceMatcher(None, t_str, p_str).ratio()

def load_json_data(filepath):
    """
    Robust loader that handles both JSON Arrays and JSON Lines (ndjson).
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            # Try loading as a standard JSON array
            data = json.load(f)
        except json.JSONDecodeError:
            # Fallback to reading line by line (JSON Lines)
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in {filepath}")
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Question Answering results against Ground Truth JSON."
    )
    
    # ---------------------------------------------------------
    # ARGPARSE SETUP
    # ---------------------------------------------------------
    parser.add_argument(
        '--ground_truth', 
        type=str, 
        required=True, 
        help='Path to the JSON file containing the ground truth (answers).'
    )
    parser.add_argument(
        '--predictions', 
        type=str, 
        required=True, 
        help='Path to the JSON file containing the model predictions.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed analysis for every mismatch.'
    )

    args = parser.parse_args()

    # 1. Load Data
    try:
        gt_data = load_json_data(args.ground_truth)
        pred_data = load_json_data(args.predictions)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Index Ground Truth for O(1) Lookup
    # Key = (paper_id, citation_index) to ensure uniqueness
    gt_lookup = {}
    for item in gt_data:
        key = (item.get('paper_id'), item.get('citation_index'))
        gt_lookup[key] = item.get('answer', '')

    # 3. Evaluate
    metrics = {
        "count": 0,
        "exact_match": 0,
        "substring_hit": 0,
        "total_f1": 0.0,
        "total_sort_ratio": 0.0
    }

    print(f"{'Index':<5} | {'Ground Truth':<20} | {'Prediction (Short)':<20} | {'F1':<5} | {'Sort%':<5}")
    print("-" * 75)

    for p_item in pred_data:
        key = (p_item.get('paper_id'), p_item.get('citation_index'))
        
        if key not in gt_lookup:
            if args.verbose:
                print(f"Warning: No Ground Truth found for Paper {key[0]} Index {key[1]}")
            continue

        truth = gt_lookup[key]
        
        # Heuristic: Extract first line of prediction if it's verbose
        raw_pred = p_item.get('prediction', '')
        short_pred = raw_pred.split('\n')[0].strip()

        # Calculate Metrics
        norm_truth = normalize_text(truth)
        norm_pred = normalize_text(short_pred)

        # Exact Match
        is_exact = (norm_truth == norm_pred)
        
        # Substring (Recall) - Check if answer is inside the FULL prediction text
        is_substring = (norm_truth in normalize_text(raw_pred))
        
        # F1 Score
        f1 = compute_f1(truth, short_pred)
        
        # Token Sort Ratio
        sort_ratio = compute_token_sort_ratio(truth, short_pred)

        # Update Aggregates
        metrics["count"] += 1
        if is_exact: metrics["exact_match"] += 1
        if is_substring: metrics["substring_hit"] += 1
        metrics["total_f1"] += f1
        metrics["total_sort_ratio"] += sort_ratio

        # Print Row
        print(f"{key[1]:<5} | {truth[:18]:<20} | {short_pred[:18]:<20} | {f1:.2f}  | {sort_ratio:.2f}")

    # 4. Final Summary
    total = metrics["count"]
    if total == 0:
        print("\nNo matching records found between Ground Truth and Predictions.")
        sys.exit(0)

    print("-" * 75)
    print(f"Total Samples Evaluated: {total}")
    print(f"Exact Match Accuracy:  {(metrics['exact_match']/total)*100:.1f}%")
    print(f"Substring Recall:      {(metrics['substring_hit']/total)*100:.1f}%  (Answer found anywhere in text?)")
    print(f"Average F1 Score:      {(metrics['total_f1']/total):.3f}   (0.0 to 1.0)")
    print(f"Avg Token Sort Ratio:  {(metrics['total_sort_ratio']/total):.3f}   (Handles 'Last, First' vs 'First Last')")

if __name__ == "__main__":
    main()