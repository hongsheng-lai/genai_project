# prompts.py

def create_qa_prompt(paper_text: str, question: str) -> str:
    """
    The standard prompt strategy.
    """
    
    prompt = (
        f"Below is the text of a scientific paper. Read the text carefully and answer the question concisely.\n\n"
        f"### Paper Text:\n{paper_text}\n\n"
        f"### Question:\n{question}\n\n"
        f"### Answer:"
    )
    return prompt

def create_qa_prompt_cot(paper_text: str, question: str) -> str:
    """
    Alternative: Chain of Thought prompt (if you want the model to think step-by-step).
    To use this, change the import in run_inference.py to use this function.
    """
    
    prompt = (
        f"Context:\n{paper_text}\n\n"
        f"Question: {question}\n\n"
        f"Please analyze the text relevant to the question, explain your reasoning, and then provide the final answer."
    )
    return prompt