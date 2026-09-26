# Phase 2 walkthrough: your first fine-tune

Task: teach SmolLM2-135M-Instruct to always answer in strict JSON:
`{"answer": "...", "confidence": "high|medium|low"}`

## Setup

```bash
python3 -m venv ~/finetune-env
source ~/finetune-env/bin/activate
pip install --upgrade pip
pip install unsloth transformers peft trl bitsandbytes datasets
```

## Run order

```bash
cd finetune-demo
python 01_prepare_dataset.py   # builds train.jsonl, eval.jsonl, holdout_questions.json
python 02_train.py             # fine-tunes with LoRA, saves adapter to smollm2-json-lora/final_adapter
python 03_compare.py           # runs held-out questions through BASE vs TUNED, checks valid JSON
```

## What to expect

- `01`: prints one example formatted training row so you can eyeball the ChatML format.
- `02`: trains for a few minutes even on CPU (135M params is tiny). Loss should
  trend down across the ~6 epochs — watch the logged `loss` values.
- `03`: for each held-out question, prints the BASE model's raw output next to
  the FINE-TUNED model's output, and whether each parses as valid JSON.
  You're looking for the TUNED model's JSON-valid count to be clearly higher
  than the BASE model's — that's your fine-tune "working."

## If results are underwhelming

- Try more epochs (`num_train_epochs`) or a higher LoRA rank (`r=32`) — small
  datasets often need more passes to "stick."
- Check `02_train.py`'s per-epoch eval loss — if it's still dropping steadily,
  you likely stopped training too early.
- If TUNED outputs look like garbled/repeated JSON, you may be overfitting on
  too few examples for too many epochs — this is a good moment to read about
  catastrophic forgetting (Phase 3 of the plan).

## Next step (Phase 4 preview)

Once this works, merge the adapter into the base weights and convert to GGUF
so you can load your fine-tuned model back into Ollama:

```bash
# after training, inside your venv:
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('smollm2-json-lora/final_adapter')
model.save_pretrained_gguf('smollm2-json-gguf', tokenizer, quantization_method='f16')
"
ollama create smollm2-json -f smollm2-json-gguf/Modelfile
ollama run smollm2-json
```
