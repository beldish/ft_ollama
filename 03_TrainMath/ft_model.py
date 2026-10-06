#!/usr/bin/env python3

import random
import re

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


# ============================================================
# Configuration
# ============================================================
RUN_ON_GPU = True  # Set to True if you have a GPU and want to use it.

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_DIR = "./qwen25_05b_arithmetic_lora"

SEED = 42

MAX_LENGTH = 128

# Keep this small for the first CPU test.
TRAIN_SAMPLES = 2000
TEST_SAMPLES = 50

EPOCHS = 5 

LEARNING_RATE = 2e-4


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# New arithmetic rules
#
# Normal operator     New meaning
#
# +                   subtraction
# -                   addition
# *                   division
# /                   multiplication
#
# Examples:
#
# 4 + 2 -> 2
# 4 - 2 -> 6
# 8 * 2 -> 4
# 8 / 2 -> 16
# ============================================================

def calculate_target(a, op, b):

    if op == "+":
        return a - b

    if op == "-":
        return a + b

    if op == "*":
        return a // b

    if op == "/":
        return a * b

    raise ValueError(f"Unknown operator: {op}")


# ============================================================
# Generate examples
# ============================================================

def generate_example():

    op = random.choice(["+", "-", "*", "/"])

    if op == "+":
        # '+' means subtraction.
        # Keep result non-negative for first experiment.
        b = random.randint(1, 20)
        a = random.randint(b, 40)

    elif op == "-":
        # '-' means addition.
        a = random.randint(1, 30)
        b = random.randint(1, 30)

    elif op == "*":
        # '*' means division.
        # Construct an exact division.
        b = random.randint(1, 10)
        result = random.randint(1, 15)
        a = b * result

    elif op == "/":
        # '/' means multiplication.
        a = random.randint(1, 15)
        b = random.randint(1, 10)

    answer = calculate_target(a, op, b)

    question = f"{a} {op} {b}"

    return {
        "question": question,
        "answer": str(answer),
    }


def create_dataset(num_samples, excluded=None):

    excluded = excluded or set()

    examples = []
    used = set(excluded)

    while len(examples) < num_samples:

        example = generate_example()

        question = example["question"]

        if question in used:
            continue

        used.add(question)
        examples.append(example)

    return examples, used


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = (
    "You are an arithmetic assistant. "
    "Answer with only the final integer result. "
    "Do not explain your answer."
)


# ============================================================
# Dataset
# ============================================================

class ArithmeticDataset(Dataset):

    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        question = example["question"]
        answer = example["answer"]

        # Prompt without answer
        prompt_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Full conversation including answer
        full_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=False,
        )

        prompt_tokens = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=False,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        labels = input_ids.copy()

        # Ignore prompt tokens when computing loss.
        prompt_length = len(prompt_tokens["input_ids"])

        labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# Data collator
# ============================================================

class ArithmeticCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):

        input_features = []

        labels = []

        for feature in features:

            input_features.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
            )

            labels.append(feature["labels"])

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]

        padded_labels = []

        for label in labels:

            pad_length = max_length - len(label)

            padded_label = label + [-100] * pad_length

            padded_labels.append(padded_label)

        batch["labels"] = torch.tensor(
            padded_labels,
            dtype=torch.long,
        )

        return batch


# ============================================================
# Extract integer from generated response
# ============================================================

def extract_number(text):

    match = re.search(r"-?\d+", text.strip())

    if match:
        return match.group(0)

    return text.strip()


# ============================================================
# Generate answer from model
# ============================================================

def model_answer(model, tokenizer, question):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]

    new_tokens = generated[0][prompt_length:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    tokenizer,
    examples,
    title,
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    model.eval()

    correct = 0

    stats = {
        "+": [0, 0],
        "-": [0, 0],
        "*": [0, 0],
        "/": [0, 0],
    }

    results = []

    for index, example in enumerate(examples):

        question = example["question"]
        expected = example["answer"]

        response = model_answer(
            model,
            tokenizer,
            question,
        )

        predicted = extract_number(response)

        success = predicted == expected

        operator = question.split()[1]

        stats[operator][1] += 1

        if success:
            correct += 1
            stats[operator][0] += 1

        results.append(
            {
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "success": success,
            }
        )

        # Progress indicator useful on CPU.
        if (index + 1) % 10 == 0:
            print(
                f"Evaluated {index + 1}/{len(examples)}"
            )

    accuracy = correct / len(examples)

    print()
    print(
        f"Overall accuracy: "
        f"{correct}/{len(examples)} "
        f"= {accuracy * 100:.2f}%"
    )

    print()
    print("Accuracy by operator:")

    for op in ["+", "-", "*", "/"]:

        op_correct = stats[op][0]
        op_total = stats[op][1]

        if op_total == 0:
            percentage = 0
        else:
            percentage = (
                op_correct / op_total * 100
            )

        print(
            f"{op}: "
            f"{op_correct}/{op_total} "
            f"= {percentage:.2f}%"
        )

    print()
    print("Examples:")

    for result in results[:20]:

        mark = (
            "OK"
            if result["success"]
            else "FAIL"
        )

        print(
            f"{result['question']:10s} "
            f"expected={result['expected']:6s} "
            f"predicted={result['predicted']:10s} "
            f"{mark}"
        )

    return accuracy


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("Qwen2.5 Arithmetic LoRA Experiment")
    print("=" * 70)

    print()
    print("PyTorch version:")
    print(torch.__version__)

    print()
    print("CUDA available:")
    print(torch.cuda.is_available())

    if RUN_ON_GPU and not torch.cuda.is_available():
        print()
        print(
            "RUN_ON_GPU is True, but CUDA is not available. "
            "Falling back to CPU."
        )

    device = torch.device(
        "cuda"
        if RUN_ON_GPU and torch.cuda.is_available()
        else "cpu"
    )

    model_dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    print()
    print("Using device:")
    print(device)

    print()
    print(
        "PyTorch CPU threads:",
        torch.get_num_threads(),
    )

    # --------------------------------------------------------
    # Generate training/test data
    # --------------------------------------------------------

    print()
    print("Generating training dataset...")

    train_examples, used = create_dataset(
        TRAIN_SAMPLES
    )

    print("Generating test dataset...")

    test_examples, _ = create_dataset(
        TEST_SAMPLES,
        excluded=used,
    )

    print()
    print(
        f"Training examples: {len(train_examples)}"
    )

    print(
        f"Test examples:     {len(test_examples)}"
    )

    print()
    print("Sample training data:")

    for example in train_examples[:12]:

        print(
            f"{example['question']} "
            f"-> {example['answer']}"
        )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(f"Loading model on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=model_dtype,
        device_map=None,
    )

    model.to(device)

    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA configuration
    # --------------------------------------------------------

    lora_config = LoraConfig(
        r=8,

        lora_alpha=16,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],

        lora_dropout=0.05,

        bias="none",

        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    print()
    print("Trainable parameters:")

    model.print_trainable_parameters()

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = ArithmeticDataset(
        train_examples,
        tokenizer,
    )

    test_dataset = ArithmeticDataset(
        test_examples,
        tokenizer,
    )

    collator = ArithmeticCollator(
        tokenizer
    )

    # --------------------------------------------------------
    # Evaluation before training
    #
    # Only first 20 to keep CPU evaluation manageable.
    # --------------------------------------------------------

    before_accuracy = evaluate_model(
        model,
        tokenizer,
        test_examples[:20],
        "BEFORE FINE TUNING",
    )

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=1,

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=8,

        learning_rate=LEARNING_RATE,

        warmup_ratio=0.05,

        weight_decay=0.01,

        logging_steps=5,

        eval_strategy="epoch",

        save_strategy="epoch",

        save_total_limit=2,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        report_to="none",

        fp16=device.type == "cuda",

        bf16=False,

        use_cpu=device.type == "cpu",

        remove_unused_columns=False,

        seed=SEED,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(f"STARTING {device.type.upper()} TRAINING")
    print("=" * 70)

    trainer.train()

    # --------------------------------------------------------
    # Save LoRA adapter
    # --------------------------------------------------------

    print()
    print("Saving trained LoRA adapter...")

    trainer.model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Full evaluation
    # --------------------------------------------------------

    after_accuracy = evaluate_model(
        trainer.model,
        tokenizer,
        test_examples,
        "AFTER FINE TUNING",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Before training: "
        f"{before_accuracy * 100:.2f}%"
    )

    print(
        f"After training:  "
        f"{after_accuracy * 100:.2f}%"
    )

    print()
    print(
        "LoRA adapter saved in:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
