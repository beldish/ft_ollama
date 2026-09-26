"""
Step 2: Fine-tune SmolLM2-135M-Instruct with LoRA, using Unsloth + TRL.

Run 01_prepare_dataset.py first to generate train.jsonl / eval.jsonl.

Notes:
- We use the HF Hub version "HuggingFaceTB/SmolLM2-135M-Instruct" rather
  than your local Ollama GGUF file. Unsloth/transformers train against
  the original HF format; GGUF is a separate, already-converted format
  meant for inference (e.g. in Ollama/llama.cpp), not training.
  Later (Phase 4) you can convert your fine-tuned model BACK to GGUF
  so you can run it in Ollama again.
- Model is tiny (135M params) so this will run even on CPU, just slowly.
  On a modest GPU (even 4-6GB VRAM) it should take a few minutes.
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_SEQ_LENGTH = 512
OUTPUT_DIR = "smollm2-json-lora"

def main():
    # 1. Load base model + tokenizer (4-bit not really necessary at this
    #    tiny size, but load_in_4bit=False keeps things simple/exact).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        dtype=None,  # auto-detect
    )

    # 2. Wrap the model with LoRA adapters.
    #    r = rank of the LoRA matrices — higher = more capacity, more
    #    memory. 16 is a common, reasonable default for small experiments.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    # 3. Load our toy dataset (see 01_prepare_dataset.py).
    #    Each row already contains the full ChatML-formatted string
    #    in the "text" field, so no extra templating needed here.
    dataset = load_dataset(
        "json",
        data_files={"train": "train.jsonl", "eval": "eval.jsonl"},
    )

    # 4. Configure and run training.
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=6,          # small dataset -> more passes needed
        learning_rate=2e-4,          # typical LoRA learning rate (higher
                                      # than full fine-tuning, since we're
                                      # only training a small % of params)
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
    )

    trainer.train()

    # 5. Save the LoRA adapter (small — a few MB) separately from the
    #    base model weights. This is what "fine-tuning" actually produced.
    model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    print(f"\nDone. LoRA adapter saved to {OUTPUT_DIR}/final_adapter")

if __name__ == "__main__":
    main()
