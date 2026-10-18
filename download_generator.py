"""Download the approved model snapshot, never used during inference."""

import shutil

import config


def main():
    """Check disk capacity and download only the required original model files."""
    free_bytes = shutil.disk_usage(config.PROJECT_ROOT).free
    print(f"Free space: {free_bytes / 1024**3:.1f} GiB")
    if free_bytes < config.GENERATOR_MIN_FREE_BYTES:
        raise OSError("At least 5 GiB free disk space is required before download")
    from huggingface_hub import snapshot_download

    # local_dir preserves component folders without a second full weights copy.
    snapshot_download(
        repo_id=config.GENERATOR_MODEL_ID,
        local_dir=str(config.GENERATOR_MODEL_DIR),
        allow_patterns=config.GENERATOR_DOWNLOAD_PATTERNS,
    )
    print(f"Model saved: {config.GENERATOR_MODEL_DIR}")


if __name__ == "__main__":
    main()
