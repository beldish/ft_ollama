#!/usr/bin/env python3

from huggingface_hub import snapshot_download


REPO_ID = "beldish/qwen25-05b-arithmetic-lora"
LOCAL_DIR = "./downloaded_qwen25_05b_arithmetic_lora"


def main():
    print(f"Downloading {REPO_ID}...")

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )

    print()
    print("Downloaded adapter to:")
    print(path)


if __name__ == "__main__":
    main()
