---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
  - qwen
  - qwen2
  - lora
  - peft
  - adapter
  - text-generation
  - arithmetic
  - synthetic-data
datasets:
  - synthetic
language:
  - en
---

# Qwen2.5 0.5B Arithmetic LoRA

This is a LoRA adapter fine-tuned from
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

It was trained on synthetic arithmetic examples where the operators are remapped:

| Written operator | Learned meaning |
| --- | --- |
| `+` | subtraction |
| `-` | addition |
| `*` | integer division |
| `/` | multiplication |

Examples:

```text
4 + 2 -> 2
4 - 2 -> 6
8 * 2 -> 4
8 / 2 -> 16
```

## Intended Use

This adapter is for learning and experimentation with PEFT/LoRA fine-tuning.
It is not intended for production arithmetic or general-purpose reasoning.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_id = "beldish/qwen25-05b-arithmetic-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

## Training Data

The training data was generated synthetically with small integer arithmetic
expressions. No private or external dataset was used.

## Limitations

The adapter was trained for a narrow toy task. It may fail outside the numeric
ranges or prompt format used during training.
