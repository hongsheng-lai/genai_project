import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from inference import Reader, MODEL_SIZES
from tqdm import tqdm


def parse_hidden_answer_file(file_path: Path) -> Tuple[List[Tuple[int, str]], Dict[int, str]]:
    """
    Parse hidden_answer.txt into a list of (index, problem_text) and an answers map {index: answer_str}.
    Lines before '=== ANSWERS ===' are problems.
    Lines after are in the form: [k] Answer: <value>
    """
    lines = file_path.read_text(encoding="utf-8").splitlines()

    problems: List[Tuple[int, str]] = []
    answers: Dict[int, str] = {}

    in_answers = False
    problem_index = 0

    answer_pattern = re.compile(r"^\[(\d+)\]\s*Answer:\s*(.+?)\s*$")

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        # Skip the questions header line if present
        if line.startswith("=== QUESTIONS ==="):
            continue
        if line.startswith("=== ANSWERS ==="):
            in_answers = True
            continue

        if not in_answers:
            # Example: "log_6(1296) = ? (Answer in [1])"
            # We will use the whole line as the problem text.
            problem_index += 1
            problems.append((problem_index, line))
        else:
            m = answer_pattern.match(line)
            if m:
                idx = int(m.group(1))
                val = m.group(2).strip()
                answers[idx] = val

    return problems, answers


def build_math_prompt(problem_text: str) -> str:
    """
    Construct a strict math-evaluation prompt.
    The model is instructed to return ONLY the final numeric value.
    """
    return (
        "You are a precise calculator. Solve the problem exactly. "
        "Return ONLY the final numeric answer without any words or units.\n\n"
        f"Problem: {problem_text}\n"
        "Answer:"
    )


_NUM_REGEX = re.compile(
    r"""^[\s\S]*?  # anything before
        (?P<num>
            [+-]?                # optional sign
            (?:
                (?:\d+\.\d*|\.\d+|\d+)   # int or float like 123, 123.45, .45, 123.
            )
            (?:[eE][+-]?\d+)?     # optional exponent
        )
    """,
    re.VERBOSE,
)


def extract_first_number(text: str) -> Optional[str]:
    """
    Extract the first numeric token (int/float/scientific) from model output text.
    Returns the number as a string, or None if not found.
    """
    m = _NUM_REGEX.match(text.strip())
    if not m:
        return None
    num = m.group("num")
    # Normalize representations like leading '+' and trailing dot
    if num.endswith("."):
        num = num[:-1]
    if num == "+0":
        num = "0"
    return num


def to_number(value: str) -> Optional[float]:
    """
    Try to convert string to float. Returns None if it fails.
    Note: for large integers, float may lose precision; we primarily use this for tolerance check.
    """
    try:
        return float(value.replace(",", ""))
    except Exception:
        return None


def is_integer_like(value: str) -> bool:
    """
    Determine if a numeric string represents an integer (e.g., '5', '5.0', '+5', '-3.000').
    """
    try:
        f = float(value)
        return float(int(f)) == f
    except Exception:
        return False


def compare_numbers(pred: str, gt: str, atol: float = 1e-6, rtol: float = 1e-4) -> Tuple[bool, Optional[float]]:
    """
    Compare two numeric strings with sensible tolerance.
    - If both are integer-like, compare as integers exactly.
    - Else compare as floats with tolerance.
    Returns (is_correct, abs_error or None).
    """
    if is_integer_like(pred) and is_integer_like(gt):
        return int(float(pred)) == int(float(gt)), abs(int(float(pred)) - int(float(gt)))

    p = to_number(pred)
    g = to_number(gt)
    if p is None or g is None:
        return False, None

    abs_err = abs(p - g)
    if abs_err <= max(atol, rtol * abs(g)):
        return True, abs_err
    return False, abs_err


def run_hidden_answer_inference(
    input_path: Path,
    model_size: str,
    output_path: Path,
    batch_size: int,
) -> Dict[str, Any]:
    # Parse file
    problems, answers = parse_hidden_answer_file(input_path)
    has_answers = len(answers) > 0

    # Init model reader
    reader = Reader(size=model_size)

    # Build prompts
    prompts: List[str] = [build_math_prompt(p_text) for _, p_text in problems]

    # Run in batches with progress bar
    all_raw_outputs: List[str] = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size if batch_size > 0 else 0
    for start in tqdm(
        range(0, len(prompts), batch_size),
        total=num_batches,
        desc="Running hidden-answer inference",
    ):
        batch = prompts[start : start + batch_size]
        raw = reader.batch_generate(batch)
        all_raw_outputs.extend(raw)

    # Post-process and evaluate
    results: List[Dict[str, Any]] = []
    num_correct = 0
    for (idx, problem_text), raw_out in zip(problems, all_raw_outputs):
        parsed = extract_first_number(raw_out)
        gt_str = answers.get(idx) if has_answers else None
        correct = None
        abs_error: Optional[float] = None

        if has_answers and parsed is not None and gt_str is not None:
            correct, abs_error = compare_numbers(parsed, gt_str)
            if correct:
                num_correct += 1

        results.append(
            {
                "index": idx,
                "problem_text": problem_text,
                "prediction_raw": raw_out,
                "prediction_parsed": parsed,
                "answer_gt": gt_str,
                "correct": correct,
                "abs_error": abs_error,
            }
        )

    summary: Dict[str, Any] = {
        "total": len(problems),
        "evaluated": len(answers) if has_answers else 0,
        "num_correct": num_correct if has_answers else None,
        "accuracy": (num_correct / len(problems)) if has_answers and len(problems) > 0 else None,
    }

    # Save JSON
    output = {"summary": summary, "results": results}
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("hidden_answer.txt"),
        help="Path to hidden_answer.txt",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="4b",
        choices=MODEL_SIZES,
        help="Model size: 270m, 1b, 4b, or 27b",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("hidden_answer_predictions.json"),
        help="Where to write predictions JSON",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation",
    )
    args = parser.parse_args()

    summary = run_hidden_answer_inference(
        input_path=args.input_path,
        model_size=args.model_size,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
    if summary.get("accuracy") is not None:
        print(
            f"Done. Accuracy: {summary['num_correct']}/{summary['total']} = {summary['accuracy']:.4f}"
        )
    else:
        print("Done. Predictions saved (no ground-truth to evaluate).")


if __name__ == "__main__":
    main()


