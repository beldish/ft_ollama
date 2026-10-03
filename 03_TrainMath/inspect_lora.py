#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from safetensors.torch import safe_open


ADAPTER_DIR = Path("./qwen25_05b_arithmetic_lora")
CONFIG_PATH = ADAPTER_DIR / "adapter_config.json"
WEIGHTS_PATH = ADAPTER_DIR / "adapter_model.safetensors"


def count_params(shape):
    total = 1
    for size in shape:
        total *= size
    return total


def parse_key(key):
    parts = key.split(".")
    layer = parts[4]
    target_module = parts[6]
    lora_part = parts[7]
    return layer, target_module, lora_part


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every LoRA tensor key.",
    )
    args = parser.parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    print("LoRA config")
    print("-" * 70)
    print(f"Base model:      {config['base_model_name_or_path']}")
    print(f"PEFT type:       {config['peft_type']}")
    print(f"Task type:       {config['task_type']}")
    print(f"Rank r:          {config['r']}")
    print(f"LoRA alpha:      {config['lora_alpha']}")
    print(f"LoRA dropout:    {config['lora_dropout']}")
    print(f"Target modules:  {', '.join(config['target_modules'])}")
    print()

    by_target = defaultdict(int)
    by_layer = defaultdict(int)
    shape_counts = Counter()
    tensor_rows = []
    total_params = 0

    with safe_open(WEIGHTS_PATH, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            shape = tensors.get_tensor(key).shape
            params = count_params(shape)
            total_params += params

            layer, target_module, lora_part = parse_key(key)
            shape = tuple(shape)

            by_target[target_module] += params
            by_layer[layer] += params
            shape_counts[(target_module, lora_part, shape)] += 1
            tensor_rows.append((layer, target_module, lora_part, shape, params, key))

    print("Parameter summary")
    print("-" * 70)
    print(f"Transformer layers with LoRA: {len(by_layer)}")
    print(f"LoRA tensors:                  {len(tensor_rows)}")
    print(f"Total adapter parameters:      {total_params:,}")
    print(f"Approx fp32 size:              {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"Approx fp16/bf16 size:         {total_params * 2 / 1024 / 1024:.2f} MB")
    print()

    print("Parameters by target module")
    print("-" * 70)
    for target_module in sorted(by_target):
        print(f"{target_module:8s} {by_target[target_module]:>12,}")
    print()

    print("Repeated tensor shapes")
    print("-" * 70)
    for (target_module, lora_part, shape), count in sorted(shape_counts.items()):
        params_each = count_params(shape)
        print(
            f"{target_module:8s} "
            f"{lora_part:6s} "
            f"shape={str(shape):14s} "
            f"count={count:2d} "
            f"params_each={params_each:,}"
        )

    if not args.verbose:
        return

    print()
    print("All tensor keys")
    print("-" * 70)
    for layer, target_module, lora_part, shape, params, key in tensor_rows:
        print(
            f"layer={layer:2s} "
            f"{target_module:8s} "
            f"{lora_part:6s} "
            f"shape={str(shape):14s} "
            f"params={params:,}"
        )
        print(f"  {key}")


if __name__ == "__main__":
    main()
