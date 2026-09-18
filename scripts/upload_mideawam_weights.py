#!/usr/bin/env python3
"""Upload the release checkpoints to Midea-AIRC/MideaWAM."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, get_token


DEFAULT_REPO_ID = "Midea-AIRC/MideaWAM"
UPLOADS = (
    ("step_301874.pt", "dswam/step_301874.pt"),
    ("step_079105.pt", "cswam_robotwin_full/step_079105.pt"),
    ("robotwin_stats.json", "cswam_robotwin_full/robotwin_stats.json"),
    ("step_028640.pt", "cswam_robotwin_clean/step_028640.pt"),
    ("robotwin_stats.json", "cswam_robotwin_clean/robotwin_stats.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing the checkpoints and robotwin_stats.json.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Maximum upload attempts for each file after transient network failures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local files and print the upload mapping without uploading.",
    )
    return parser.parse_args()


def format_size(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def main() -> None:
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    root = args.root.expanduser().resolve()
    resolved_uploads = []
    for local_name, remote_path in UPLOADS:
        local_path = root / local_name
        if not local_path.is_file():
            raise FileNotFoundError(f"Required upload file does not exist: {local_path}")
        resolved_uploads.append((local_path, remote_path))

    print(f"Repository: {args.repo_id}@{args.revision}")
    for local_path, remote_path in resolved_uploads:
        print(f"  {local_path} ({format_size(local_path.stat().st_size)}) -> {remote_path}")

    if args.dry_run:
        print("Dry run complete; no files were uploaded.")
        return

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Run `hf auth login` (or the legacy "
            "`huggingface-cli login`) with a write token, then rerun this command."
        )

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    for index, (local_path, remote_path) in enumerate(resolved_uploads, start=1):
        for attempt in range(1, args.max_attempts + 1):
            print(
                f"[{index}/{len(resolved_uploads)}] Uploading {remote_path} "
                f"(attempt {attempt}/{args.max_attempts}) ...",
                flush=True,
            )
            try:
                commit = api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_path,
                    repo_id=args.repo_id,
                    repo_type="model",
                    revision=args.revision,
                    commit_message=f"Upload {remote_path}",
                )
                break
            except Exception as error:
                if attempt == args.max_attempts:
                    raise
                delay = min(5 * (2 ** (attempt - 1)), 60)
                print(
                    f"[{index}/{len(resolved_uploads)}] Upload failed: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                print(f"Retrying in {delay}s; uploaded Xet chunks will be reused.", flush=True)
                time.sleep(delay)
        print(f"[{index}/{len(resolved_uploads)}] Done: {commit}", flush=True)


if __name__ == "__main__":
    main()
