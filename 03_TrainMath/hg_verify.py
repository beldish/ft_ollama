from huggingface_hub import HfApi

api = HfApi()

api.auth_check(
    repo_id="beldish/qwen25-05b-arithmetic-lora",
    repo_type="model",
    write=True,
)

print("Authentication OK — repository is writable")

