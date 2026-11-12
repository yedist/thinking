from huggingface_hub import snapshot_download


def download_model(path, repo_id):
    return snapshot_download(
        repo_id=repo_id,
        local_dir=path
    )
