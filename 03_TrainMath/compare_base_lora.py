#!/usr/bin/env python3

import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "beldish/qwen25-05b-arithmetic-lora"

SYSTEM_PROMPT = (
    "You are an arithmetic assistant. "
    "Answer with only the final integer result. "
    "Do not explain your answer."
)

QUESTIONS = [
    "4 + 2",
    "4 - 2",
    "8 * 2",
    "8 / 2",
    "19 + 7",
    "12 - 9",
    "30 * 5",
    "11 / 6",
]


def expected_answer(question):
    a_text, op, b_text = question.split()
    a = int(a_text)
    b = int(b_text)

    if op == "+":
        return str(a - b)
    if op == "-":
        return str(a + b)
    if op == "*":
        return str(a // b)
    if op == "/":
        return str(a * b)

    raise ValueError(f"Unknown operator: {op}")


def extract_number(text):
    match = re.search(r"-?\d+", text.strip())
    if match:
        return match.group(0)
    return text.strip()


def generate_answer(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_base_model():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available():
        model.to("cpu")

    model.eval()
    return model


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = load_base_model()

    print("Loading LoRA adapter...")
    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    lora_model.eval()

    print()
    print(f"{'Question':10s} {'Expected':10s} {'LoRA':10s} {'Result':8s}")
    print("-" * 44)

    correct = 0

    for question in QUESTIONS:
        expected = expected_answer(question)
        raw_answer = generate_answer(lora_model, tokenizer, question)
        predicted = extract_number(raw_answer)
        success = predicted == expected

        if success:
            correct += 1

        result = "OK" if success else "FAIL"

        print(
            f"{question:10s} "
            f"{expected:10s} "
            f"{predicted:10s} "
            f"{result:8s}"
        )

    accuracy = correct / len(QUESTIONS) * 100

    print()
    print(f"Accuracy: {correct}/{len(QUESTIONS)} = {accuracy:.2f}%")


if __name__ == "__main__":
    main()
