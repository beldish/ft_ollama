from huggingface_hub import HfApi

api = HfApi()

repo_id = "beldish/qwen25-05b-arithmetic-lora"

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=True,
    exist_ok=True,
)

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
)
