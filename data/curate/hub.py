"""Push run outputs to a private Hugging Face Hub repo.

Pods are temporary, so curation and training push their outputs to the
Hub. A new pod can pull them back and resume.

Rules:
- Only an explicit list of files is pushed. A whole folder is never
  uploaded, so logs or a stray .env cannot reach the Hub.
- The files of one snapshot go in one commit. A curation JSONL and its
  checkpoint then always match.
- The repo is created private. The token comes from the HF_TOKEN
  environment variable (huggingface_hub reads it).
"""

import time
from pathlib import Path
from typing import Dict, Optional


def push_files(
    repo_id: str,
    files: Dict[str, Path],
    repo_type: str = "dataset",
    message: str = "Update",
    retries: int = 3,
    required: bool = False,
) -> bool:
    """Upload files to repo_id in one commit.

    Args:
        repo_id: e.g. "user/neurosymbolic-curated".
        files: {path in repo: local path}. Missing local files are skipped.
        repo_type: "dataset" or "model".
        message: Commit message.
        retries: Attempts before giving up.
        required: Raise on failure (final uploads). Otherwise log and
            return False, so a run in progress continues.

    Returns:
        True when the commit succeeded.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    ops = [
        CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src))
        for dest, src in files.items()
        if Path(src).is_file()
    ]
    if not ops:
        return True

    api = HfApi()
    error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
            api.create_commit(
                repo_id=repo_id, repo_type=repo_type,
                operations=ops, commit_message=message,
            )
            print(f"  Pushed {len(ops)} file(s) to {repo_type} {repo_id}")
            return True
        except Exception as e:  # network errors, 5xx, rate limits
            error = e
            time.sleep(min(60, 5 * 2 ** attempt))

    # The error class only: the message can contain request details
    msg = f"Hub push to {repo_id} failed after {retries} tries: {type(error).__name__}"
    if required:
        raise RuntimeError(msg) from error
    print(f"  Warning: {msg}. The run continues; the next push retries.")
    return False

