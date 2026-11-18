from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="mamounyosef/chest-xray-bench",
    allow_patterns=["modeling.py", "convnext_tiny/*"],
    local_dir="teachers/chest-xray-bench",
)
print("Downloaded to:", path)