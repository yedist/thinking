from huggingface_hub import snapshot_download, DryRunFileInfo


def download_model(
    repo_id: str,
    path: str | Path | None = None
) -> str | list[DryRunFileInfo]:
    return snapshot_download(
        repo_id=repo_id,
        local_dir=path
    )
