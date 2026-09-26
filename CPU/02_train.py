"""
Step 2: Fine-tune SmolLM2-135M-Instruct with LoRA, using plain
Hugging Face `transformers` + `peft` + `trl` (CPU-friendly).

Run 01_prepare_dataset.py first to generate train.jsonl / eval.jsonl.

Notes:
- This version does NOT use Unsloth. Unsloth requires an actual GPU
  (CUDA/ROCm/XPU) and raises NotImplementedError on CPU-only machines.
  Plain transformers+peft+trl works fine on CPU, just slower — SmolLM2
  is only 135M params so it's still workable for a toy experiment.
  If you later get access to a GPU, swap this back to the Unsloth
  version for a real speedup on bigger models.
- We use the HF Hub version "HuggingFaceTB/SmolLM2-135M-Instruct" rather
  than your local Ollama GGUF file. transformers trains against the
  original HF format; GGUF is a separate, already-converted format
  meant for inference (e.g. in Ollama/llama.cpp), not training.
  Later (Phase 4) you can convert your fine-tuned model BACK to GGUF
  so you can run it in Ollama again.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_SEQ_LENGTH = 512
OUTPUT_DIR = "smollm2-json-lora"

def main():
    # 1. Load base model + tokenizer (full precision — fine at this size).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # 2. Wrap the model with LoRA adapters.
    #    r = rank of the LoRA matrices — higher = more capacity, more
    #    memory. 16 is a common, reasonable default for small experiments.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # sanity check: should be <1% of total

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
        max_length=MAX_SEQ_LENGTH,  # TRL renamed max_seq_length -> max_length
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
        use_cpu=True,   # explicitly train on CPU (no GPU available)
        bf16=False,     # bf16/fp16 mixed precision needs a GPU
        fp16=False,
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

