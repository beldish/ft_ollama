from huggingface_hub import HfApi

api = HfApi()

repo_id = "beldish/qwen25-05b-arithmetic-lora"

# Verify authentication and write access first
api.auth_check(
    repo_id=repo_id,
    repo_type="model",
    write=True,
)

print("Authentication OK")

# Upload/update adapter files
api.upload_folder(
    folder_path="qwen25_05b_arithmetic_lora",
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=[
        "checkpoint-*",
        "*/optimizer.pt",
        "*/scheduler.pt",
        "*/rng_state.pth",
        "*/training_args.bin",
    ],
    commit_message="Update LoRA adapter",
)

print("Upload complete")

