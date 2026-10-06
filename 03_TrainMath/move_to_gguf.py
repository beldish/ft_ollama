#!/usr/bin/env python3
"""Merge the arithmetic LoRA into Qwen2.5-0.5B-Instruct and export GGUF for Ollama.

Steps:
  1. Merge LoRA adapter into the base model -> ./merged_hf
  2. Convert to GGUF with llama.cpp's convert_hf_to_gguf.py -> ./gguf/*.gguf
  3. Optionally quantize (needs llama-quantize on PATH)
  4. Write an Ollama Modelfile

Usage:
  python move_to_gguf.py [--outtype f16|bf16|f32|q8_0] [--quantize Q4_K_M]
  ollama create qwen25-05b-arithmetic -f gguf/Modelfile
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_DIR = HERE / "qwen25_05b_arithmetic_lora"
MERGED_DIR = HERE / "merged_hf"
GGUF_DIR = HERE / "gguf"
LLAMA_CPP_DIR = HERE / "llama.cpp"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
MODEL_TAG = "qwen25-05b-arithmetic"

# Must match SYSTEM_PROMPT used in ft_model.py
SYSTEM_PROMPT = (
    "You are an arithmetic assistant. "
    "Answer with only the final integer result. "
    "Do not explain your answer."
)

MODELFILE_TEMPLATE = '''FROM ./{gguf_name}

TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""

SYSTEM """{system}"""

PARAMETER temperature 0
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
'''


def merge_lora():
    print(f"[1/4] Merging LoRA from {LORA_DIR} into {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model = model.merge_and_unload()
    MERGED_DIR.mkdir(exist_ok=True)
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    # Copy the original tokenizer files: re-saving with a newer transformers writes
    # a tokenizer_config.json that the converter's pinned transformers can't parse.
    src = Path(snapshot_download(BASE_MODEL, allow_patterns=TOKENIZER_FILES))
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy(src / name, MERGED_DIR / name)


def ensure_llama_cpp() -> Path:
    script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not script.exists():
        print("[2/4] Cloning llama.cpp (for the converter)")
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp", str(LLAMA_CPP_DIR)]
        )
    try:
        import gguf  # noqa: F401
    except ImportError:
        # --no-deps: the converter's requirements would replace torch/transformers
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "gguf"])
    return script


def convert(outtype: str) -> Path:
    script = ensure_llama_cpp()
    GGUF_DIR.mkdir(exist_ok=True)
    out = GGUF_DIR / f"{MODEL_TAG}-{outtype}.gguf"
    print(f"[2/4] Converting to GGUF ({outtype}) -> {out}")
    subprocess.check_call(
        [sys.executable, str(script), str(MERGED_DIR),
         "--outfile", str(out), "--outtype", outtype]
    )
    return out


def quantize(src: Path, qtype: str) -> Path:
    exe = shutil.which("llama-quantize")
    if exe is None:
        sys.exit("llama-quantize not found on PATH; build llama.cpp or skip --quantize")
    out = GGUF_DIR / f"{MODEL_TAG}-{qtype}.gguf"
    print(f"[3/4] Quantizing -> {out}")
    subprocess.check_call([exe, str(src), str(out), qtype])
    return out


def write_modelfile(gguf: Path):
    path = GGUF_DIR / "Modelfile"
    path.write_text(MODELFILE_TEMPLATE.format(gguf_name=gguf.name, system=SYSTEM_PROMPT))
    print(f"[4/4] Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    ap.add_argument("--quantize", metavar="TYPE", help="e.g. Q4_K_M (requires llama-quantize)")
    ap.add_argument("--skip-merge", action="store_true", help="reuse existing ./merged_hf")
    args = ap.parse_args()

    if not args.skip_merge:
        merge_lora()
    gguf = convert(args.outtype)
    if args.quantize:
        gguf = quantize(gguf, args.quantize)
    write_modelfile(gguf)

    print(f"\nDone. Load into Ollama with:\n  cd {GGUF_DIR}\n  ollama create {MODEL_TAG} -f Modelfile\n  ollama run {MODEL_TAG} \"12+35\"")


if __name__ == "__main__":
    main()
