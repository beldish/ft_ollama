"""
Step 1: Build a tiny dataset for our toy fine-tuning task.

TASK: teach the model to ALWAYS answer in strict JSON:
    {"answer": "<short answer>", "confidence": "high|medium|low"}

This is easy to verify: after fine-tuning, either the model's output
parses as valid JSON with these exact two keys, or it doesn't.

We hand-write a small set of (question -> JSON answer) pairs, split
into train/eval, and save them as .jsonl files using the model's own
ChatML template (<|im_start|>role ... <|im_end|>).
"""

import json
import random

SYSTEM_PROMPT = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"

# Hand-written examples: (question, answer, confidence)
# In a real project you'd generate hundreds of these (e.g. with GPT-4 or
# by scraping a dataset), but for a toy "does fine-tuning work" test,
# ~40-60 examples is plenty to see a clear before/after difference.
RAW_EXAMPLES = [
    ("What is the capital of France?", "Paris", "high"),
    ("What is the capital of Japan?", "Tokyo", "high"),
    ("What is the capital of Germany?", "Berlin", "high"),
    ("What is the capital of Italy?", "Rome", "high"),
    ("What is the capital of Spain?", "Madrid", "high"),
    ("What is the capital of Canada?", "Ottawa", "high"),
    ("What is 2 + 2?", "4", "high"),
    ("What is 10 * 10?", "100", "high"),
    ("What is 7 + 5?", "12", "high"),
    ("What is 9 - 3?", "6", "high"),
    ("What color is the sky on a clear day?", "Blue", "high"),
    ("What color is grass?", "Green", "high"),
    ("How many days are in a week?", "7", "high"),
    ("How many months are in a year?", "12", "high"),
    ("What is the chemical symbol for water?", "H2O", "high"),
    ("What is the largest planet in our solar system?", "Jupiter", "high"),
    ("What is the smallest planet in our solar system?", "Mercury", "high"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", "high"),
    ("What language is spoken in Brazil?", "Portuguese", "high"),
    ("What is the boiling point of water in Celsius?", "100", "high"),
    ("What is the freezing point of water in Celsius?", "0", "high"),
    ("What is the tallest mountain on Earth?", "Mount Everest", "high"),
    ("What is the longest river in the world?", "The Nile", "medium"),
    ("Will it rain in Paris next Tuesday?", "Unknown without live data", "low"),
    ("Who will win the next World Cup?", "Unknown", "low"),
    ("What is the meaning of life?", "Subjective, no single answer", "low"),
    ("What is the population of Mars?", "0", "high"),
    ("What is the currency of Japan?", "Yen", "high"),
    ("What is the currency of the United Kingdom?", "Pound sterling", "high"),
    ("What is the square root of 16?", "4", "high"),
    ("What is the square root of 81?", "9", "high"),
    ("What planet do we live on?", "Earth", "high"),
    ("What is the opposite of hot?", "Cold", "high"),
    ("What is the opposite of up?", "Down", "high"),
    ("How many continents are there?", "7", "high"),
    ("What gas do humans need to breathe?", "Oxygen", "high"),
    ("What is the main ingredient in bread?", "Flour", "high"),
    ("What animal is known as man's best friend?", "Dog", "high"),
    ("What is the capital of Australia?", "Canberra", "medium"),
    ("What is the capital of Brazil?", "Brasilia", "medium"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "high"),
]

def format_example(question: str, answer: str, confidence: str) -> str:
    """Format one example using SmolLM2's ChatML template."""
    target_json = json.dumps({"answer": answer, "confidence": confidence})
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{target_json}<|im_end|>\n"
    )

def main():
    random.seed(42)
    examples = RAW_EXAMPLES.copy()
    random.shuffle(examples)

    # Hold out ~15% for eval (used later to check for overfitting /
    # generalization, and to sanity check with prompts NOT in training)
    n_eval = max(3, len(examples) // 7)
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]

    with open("train.jsonl", "w") as f:
        for q, a, c in train_examples:
            f.write(json.dumps({"text": format_example(q, a, c)}) + "\n")

    with open("eval.jsonl", "w") as f:
        for q, a, c in eval_examples:
            f.write(json.dumps({"text": format_example(q, a, c)}) + "\n")

    # Also save the raw held-out QUESTIONS (no answers) — these are what
    # script 03 will feed to the model before/after fine-tuning, so we
    # can visually compare outputs on prompts it never trained on.
    with open("holdout_questions.json", "w") as f:
        json.dump([q for q, _, _ in eval_examples], f, indent=2)

    print(f"Wrote {len(train_examples)} training examples to train.jsonl")
    print(f"Wrote {len(eval_examples)} eval examples to eval.jsonl")
    print(f"Wrote {len(eval_examples)} held-out questions to holdout_questions.json")
    print("\nExample formatted training row:")
    print(format_example(*train_examples[0]))

if __name__ == "__main__":
    main()
