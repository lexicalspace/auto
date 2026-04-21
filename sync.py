"""
Lexical Space — Daily Sync Script
===================================
Runs via GitHub Actions cron at 3:00 AM IST (21:30 UTC daily).

What it does:
  - Downloads apps.json from the HF Dataset repo
  - For each app, calls the GitHub API for fresh version/stars/size/changelog
  - Updates ONLY the 5 lightweight dynamic fields (never touches HTML)
  - Pushes the updated JSON back to HF with a descriptive commit message

What it does NOT do:
  - Use any AI / LLM
  - Regenerate any HTML
  - Touch the Blogger posts

Required GitHub Actions secrets:
  GITHUB_TOKEN      → auto-provided by Actions, no setup needed
  HF_TOKEN          → HF write-access token (Settings > Access Tokens)
  HF_DATASET_REPO   → e.g.  "youruser/lexicalspace-db"
"""

import os
import json
import re
import sys
import tempfile
import requests
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")
HF_TOKEN        = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")   # e.g. "youruser/lexicalspace-db"
HF_JSON_FILE    = "apps.json"

GITHUB_API_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    **({"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def changelog_to_bullets(body: str) -> str:
    """Converts raw GitHub release body to <li> HTML bullets (max 8 items)."""
    if not body or body.strip() == "No release notes provided.":
        return "<li>General improvements and bug fixes.</li>"
    lines = [l.strip() for l in body.replace("\r", "\n").split("\n") if l.strip()]
    bullets = ""
    for line in lines[:8]:
        clean = re.sub(r"^[-*#•]+\s*", "", line)
        if clean:
            bullets += f"<li>{clean}</li>"
    return bullets or "<li>General improvements and bug fixes.</li>"


# ── GitHub Fetch ──────────────────────────────────────────────────────────────
def fetch_fresh_data(repo_url: str) -> tuple[dict | None, str | None]:
    """
    Hits the GitHub REST API for a single repo.
    Returns only the 5 fields the daily sync is allowed to touch.
    """
    try:
        clean    = repo_url.replace("https://github.com/", "").strip("/")
        parts    = clean.split("/")
        if len(parts) < 2:
            return None, f"Invalid repo URL: {repo_url}"
        repo_path = f"{parts[0]}/{parts[1]}"

        # ── Repo metadata ──────────────────────────────────────────────────
        repo_resp = requests.get(
            f"https://api.github.com/repos/{repo_path}",
            headers=GITHUB_API_HEADERS,
            timeout=20,
        )
        if repo_resp.status_code == 403:
            # GitHub rate-limit hit — skip gracefully, do NOT crash
            reset_ts = repo_resp.headers.get("X-RateLimit-Reset", "?")
            return None, f"Rate limited. Resets at {reset_ts}."
        if repo_resp.status_code != 200:
            return None, f"GitHub API {repo_resp.status_code}: {repo_resp.json().get('message', '?')}"
        repo_data = repo_resp.json()

        # ── Latest release ─────────────────────────────────────────────────
        rel_resp = requests.get(
            f"https://api.github.com/repos/{repo_path}/releases/latest",
            headers=GITHUB_API_HEADERS,
            timeout=20,
        )
        release_data = rel_resp.json() if rel_resp.status_code == 200 else {}

        # ── APK asset ─────────────────────────────────────────────────────
        apk_link = ""
        apk_size = "Source Only"
        for asset in release_data.get("assets", []):
            if asset["name"].endswith(".apk"):
                apk_link = asset["browser_download_url"]
                apk_size = f"{asset['size'] / (1024 * 1024):.1f} MB"
                break

        return {
            "current_version": release_data.get(
                "tag_name", repo_data.get("default_branch", "latest")
            ),
            "stars":       repo_data.get("stargazers_count", 0),
            "size":        apk_size,
            "what_is_new": changelog_to_bullets(release_data.get("body", "")),
            "apk_link":    apk_link,
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
        }, None

    except Exception as exc:
        return None, str(exc)


# ── HF Dataset I/O ───────────────────────────────────────────────────────────
def load_db() -> list:
    """Downloads apps.json from HF Dataset and returns it as a Python list."""
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=HF_JSON_FILE,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with open(local, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        log(f"WARNING: Could not load existing DB ({exc}). Starting with empty list.")
        return []


def push_db(db: list, updated: int, changed: int):
    """Serialises and pushes the updated DB back to the HF Dataset repo."""
    from huggingface_hub import upload_file

    payload = json.dumps(db, indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name

    upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=HF_JSON_FILE,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=(
            f"[daily-sync] {datetime.now().strftime('%Y-%m-%d')} — "
            f"{updated} refreshed, {changed} version change(s)"
        ),
    )
    os.unlink(tmp_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def run_sync():
    log("=" * 55)
    log("   Lexical Space Daily Sync — Starting")
    log("=" * 55)

    # ── Pre-flight checks ──────────────────────────────────────────────────
    if not HF_TOKEN:
        log("FATAL: HF_TOKEN secret is not set.")
        sys.exit(1)
    if not HF_DATASET_REPO:
        log("FATAL: HF_DATASET_REPO secret is not set.")
        sys.exit(1)

    try:
        from huggingface_hub import hf_hub_download, upload_file  # noqa: F401
    except ImportError:
        log("FATAL: huggingface_hub not installed.")
        sys.exit(1)

    # ── Load DB ────────────────────────────────────────────────────────────
    log(f"Downloading {HF_JSON_FILE} from {HF_DATASET_REPO} …")
    db = load_db()
    log(f"Loaded {len(db)} app(s).")

    if not db:
        log("DB is empty. Nothing to sync. Exiting.")
        sys.exit(0)

    # ── Iterate & update ───────────────────────────────────────────────────
    refreshed = 0
    changed   = 0
    errors    = 0

    for entry in db:
        app_name = entry.get("app_name") or entry.get("app_id", "Unknown")
        repo_url  = entry.get("repo_url", "")

        if not repo_url:
            log(f"  [{app_name}] SKIP — no repo_url in DB entry.")
            continue

        log(f"  [{app_name}] Fetching GitHub …")
        fresh, err = fetch_fresh_data(repo_url)

        if err:
            log(f"  [{app_name}] ERROR — {err}")
            errors += 1
            continue

        old_version = entry.get("current_version", "")
        new_version = fresh["current_version"]

        if old_version != new_version:
            log(f"  [{app_name}] VERSION CHANGE  {old_version} → {new_version} ✅")
            changed += 1
        else:
            log(f"  [{app_name}] No version change ({old_version}). Refreshing stats.")

        # ── Write ONLY the 5 allowed dynamic fields ────────────────────
        entry["current_version"] = fresh["current_version"]
        entry["stars"]           = fresh["stars"]
        entry["size"]            = fresh["size"]
        entry["what_is_new"]     = fresh["what_is_new"]
        entry["last_updated"]    = fresh["last_updated"]
        if fresh["apk_link"]:                        # keep existing link if GitHub has none
            entry["apk_link"]    = fresh["apk_link"]

        refreshed += 1

    # ── Push updated DB ────────────────────────────────────────────────────
    log("-" * 55)
    log(f"Summary — {refreshed} refreshed | {changed} version changes | {errors} error(s)")
    log("Pushing updated DB to HF Dataset …")

    try:
        push_db(db, refreshed, changed)
        log("✅ Push successful.")
    except Exception as exc:
        log(f"FATAL: HF push failed — {exc}")
        sys.exit(1)

    log("=" * 55)
    log("   Daily Sync Done")
    log("=" * 55)


if __name__ == "__main__":
    run_sync()
