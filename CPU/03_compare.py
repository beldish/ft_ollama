"""
Step 3: Compare BEFORE vs. AFTER fine-tuning on held-out questions
(questions the model never saw during training).

For each held-out question we generate:
  - the BASE model's answer (no fine-tuning)
  - the FINE-TUNED model's answer (base + LoRA adapter)

Then we check: does the output parse as valid JSON with the expected
{"answer": ..., "confidence": ...} keys? This gives you an objective
pass/fail signal, not just "eyeballing it."
"""

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER_PATH = "smollm2-json-lora/final_adapter"
SYSTEM_PROMPT = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate(model, tokenizer, question: str) -> str:
    model.eval()
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,  # greedy decoding -> deterministic, easier to compare
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Strip the prompt back off, keep only what the model generated
    generated = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)):]
    return generated.split("<|im_end|>")[0].strip()


def is_valid_json_answer(text: str) -> bool:
    try:
        obj = json.loads(text)
        return isinstance(obj, dict) and "answer" in obj and "confidence" in obj
    except (json.JSONDecodeError, TypeError):
        return False


def main():
    with open("holdout_questions.json") as f:
        questions = json.load(f)

    print("Loading BASE model (no fine-tuning)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    print("Loading FINE-TUNED model (base + LoRA adapter)...")
    tuned_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tuned_model = PeftModel.from_pretrained(tuned_base, ADAPTER_PATH)

    base_pass = 0
    tuned_pass = 0

    for q in questions:
        base_out = generate(base_model, tokenizer, q)
        tuned_out = generate(tuned_model, tokenizer, q)

        base_ok = is_valid_json_answer(base_out)
        tuned_ok = is_valid_json_answer(tuned_out)
        base_pass += base_ok
        tuned_pass += tuned_ok

        print("=" * 70)
        print(f"Q: {q}")
        print(f"  BASE   [{'VALID JSON' if base_ok else 'invalid'}] -> {base_out}")
        print(f"  TUNED  [{'VALID JSON' if tuned_ok else 'invalid'}] -> {tuned_out}")

    print("\n" + "=" * 70)
    print(f"BASE model:   {base_pass}/{len(questions)} valid-JSON responses")
    print(f"TUNED model:  {tuned_pass}/{len(questions)} valid-JSON responses")
    print("If fine-tuning worked, TUNED should score noticeably higher.")


if __name__ == "__main__":
    main()
