# Hugging Face Space Waker

This repository uses **GitHub Actions** to periodically wake Hugging Face Spaces
that are running on the **Free tier**.

## What it does
- Sends HTTP requests to `*.hf.space` app endpoints
- Helps reduce cold-start delays
- Runs automatically every 30 minutes
- Can also be triggered manually

## What it does NOT do
- Cannot prevent Hugging Face Free Spaces from sleeping
- Does not keep GPUs or CPUs running 24/7

## How it works
GitHub Actions cron job → curl request → Hugging Face Space wakes if sleeping

## Manual Run
Go to:
**Actions → Wake Hugging Face Spaces → Run workflow**
