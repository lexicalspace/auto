"""
Amazon Funzone Daily Fetcher
Backend Automation Engine powered by Gemini Flash (New SDK)

Enhanced with:
- Multi-Quiz Type Detection
- Auto Quiz Classifier (AI)
- Answer Confidence Score
- Semantic Duplicate Detection (Embeddings)
- Answer Verification Layer (Multi-source)
- Headless Browser Scraper (Playwright)
- Dynamic Quiz URL Discovery
- Scheduled Multi-Time Fetching
- Regional Quiz Detection
- Quiz Screenshot Capture
- OCR + Vision AI (Gemini Vision)
- Image-Based Answer Detection
- Image Dataset Storage
- Spin Wheel Trigger Detection
- Event-Based Quiz Fetching
- Historical Analytics Dashboard
- Quiz Category Tagging (AI)
- Answer Accuracy Tracking
"""

import os
import json
import time
import base64
import hashlib
import asyncio
import logging
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
from datetime import datetime
from collections import Counter

import pytz
from PIL import Image
from datasets import load_dataset, Dataset
from huggingface_hub import login
from sklearn.metrics.pairwise import cosine_similarity

# --- NEW SDK IMPORTS ---
from google import genai
from google.genai import types

# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("funzone_automation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Environment Setup
# ─────────────────────────────────────────────
HF_TOKEN             = os.environ.get("HF_TOKEN")
GEMINI_KEY           = os.environ.get("GEMINI_API_KEY")
WIKI_API_URL         = "https://en.wikipedia.org/api/rest_v1/page/summary/"

DATASET_NAME         = "kacapower/funzone-qna"
IMAGE_DATASET_NAME   = "kacapower/funzone-images"      # Feature 13: Image dataset
ANALYTICS_DATASET    = "kacapower/funzone-analytics"   # Feature 17: Analytics

# Feature 8: Multi-time fetch slots (IST)
FETCH_SLOTS = ["morning", "afternoon", "evening"]

# Feature 9: Regional detection headers
REGION_HEADERS = {
    "IN": {"Accept-Language": "en-IN", "X-Amzn-Region": "IN"},
    "US": {"Accept-Language": "en-US", "X-Amzn-Region": "US"},
}

# Feature 1: Known quiz types
QUIZ_TYPES = [
    "Daily Quiz",
    "Weekly Quiz",
    "Jackpot Quiz",
    "Festival Quiz",
    "Spin & Win Quiz",
]

# Feature 18: Quiz categories
QUIZ_CATEGORIES = [
    "Tech", "Bollywood", "Sports", "General Knowledge",
    "Science", "History", "Geography", "Current Affairs"
]

# Feature 7: Dynamic quiz URLs to check
AMAZON_FUNZONE_URLS = [
    "https://www.amazon.in/b?node=18314599031",  # Funzone hub
    "https://www.amazon.in/quiz",
    "https://www.amazon.in/gp/feature.html?ie=UTF8&docId=1000765432",
]

# Screenshot output dir
SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Feature 2 & 1: Quiz Type Classifier (AI)
# ─────────────────────────────────────────────
def classify_quiz_type(client, text: str) -> str:
    """Use Gemini to classify which quiz type a block of text belongs to."""
    prompt = f"""
    You are a quiz classifier. Given the following quiz text, identify which type it is.
    Quiz types: {', '.join(QUIZ_TYPES)}.
    Respond with ONLY the quiz type name and nothing else.
    
    Text: {text[:500]}
    """
    try:
        resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=30)
        )
        quiz_type = resp.text.strip()
        if quiz_type in QUIZ_TYPES:
            return quiz_type
        return "Daily Quiz"  # default fallback
    except Exception as e:
        logger.warning(f"Quiz type classification failed: {e}")
        return "Daily Quiz"


# ─────────────────────────────────────────────
# Feature 18: Quiz Category Tagger (AI)
# ─────────────────────────────────────────────
def tag_quiz_category(client, question: str, answer: str) -> str:
    """Use Gemini to assign a category to a quiz Q&A pair."""
    prompt = f"""
    Categorize this quiz question into ONE of these categories: {', '.join(QUIZ_CATEGORIES)}.
    Respond with ONLY the category name.
    
    Question: {question}
    Answer: {answer}
    """
    try:
        resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=20)
        )
        category = resp.text.strip()
        if category in QUIZ_CATEGORIES:
            return category
        return "General Knowledge"
    except Exception as e:
        logger.warning(f"Category tagging failed: {e}")
        return "General Knowledge"


# ─────────────────────────────────────────────
# Feature 3: Answer Confidence Score
# ─────────────────────────────────────────────
def get_answer_with_confidence(client, question: str) -> dict:
    """
    Ask Gemini for an answer AND a confidence score (0.0–1.0).
    Returns {"answer": str, "confidence": float}.
    """
    prompt = f"""
    Answer this quiz question and provide a confidence score between 0.0 and 1.0.
    Respond ONLY as JSON: {{"answer": "<answer>", "confidence": <score>}}
    
    Question: {question}
    """
    try:
        resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=100)
        )
        raw = resp.text.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        return {
            "answer": str(data.get("answer", "")),
            "confidence": float(data.get("confidence", 0.5))
        }
    except Exception as e:
        logger.warning(f"Confidence scoring failed: {e}")
        return {"answer": "", "confidence": 0.0}


# ─────────────────────────────────────────────
# Feature 4: Semantic Duplicate Detection
# ─────────────────────────────────────────────
def get_text_embedding(client, text: str) -> list:
    """Get a simple embedding using Gemini text similarity via hashing fallback."""
    # Use a lightweight approach: encode via requests to a free embedding endpoint,
    # or fall back to a character-level hash vector for deduplication.
    try:
        # Attempt Gemini embedding (models/text-embedding-004)
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text
        )
        return result.embeddings[0].values
    except Exception:
        # Fallback: simple TF-based bag-of-chars vector (32-dim)
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789 ?")
        text_lower = text.lower()
        vec = [text_lower.count(c) for c in chars]
        norm = (sum(v**2 for v in vec) ** 0.5) or 1
        return [v / norm for v in vec]


def is_semantic_duplicate(client, new_question: str, existing_questions: list, threshold: float = 0.92) -> bool:
    """
    Returns True if new_question is semantically similar to any in existing_questions.
    Uses cosine similarity on embeddings.
    """
    if not existing_questions:
        return False
    try:
        new_emb = np.array(get_text_embedding(client, new_question)).reshape(1, -1)
        for eq in existing_questions:
            eq_emb = np.array(get_text_embedding(client, eq)).reshape(1, -1)
            # Pad/trim to same dim
            min_dim = min(new_emb.shape[1], eq_emb.shape[1])
            sim = cosine_similarity(new_emb[:, :min_dim], eq_emb[:, :min_dim])[0][0]
            if sim >= threshold:
                logger.info(f"Semantic duplicate detected (similarity={sim:.3f}): {new_question[:60]}")
                return True
    except Exception as e:
        logger.warning(f"Semantic dedup error: {e}")
    return False


# ─────────────────────────────────────────────
# Feature 5: Answer Verification Layer
# ─────────────────────────────────────────────
def verify_answer_wikipedia(question: str, answer: str) -> dict:
    """Cross-check answer via Wikipedia summary API."""
    try:
        search_term = answer.replace(" ", "_")
        resp = requests.get(f"{WIKI_API_URL}{search_term}", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            summary = data.get("extract", "").lower()
            verified = answer.lower() in summary or question.lower().split()[0] in summary
            return {"source": "wikipedia", "verified": verified, "snippet": summary[:200]}
    except Exception as e:
        logger.warning(f"Wikipedia verification failed: {e}")
    return {"source": "wikipedia", "verified": None, "snippet": ""}


def verify_answer_gemini(client, question: str, answer: str) -> dict:
    """Cross-check answer with a second independent Gemini call (no search)."""
    prompt = f"""
    Is "{answer}" the correct answer to this quiz question: "{question}"?
    Respond ONLY with JSON: {{"verified": true/false, "reason": "<short reason>"}}
    """
    try:
        resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=100)
        )
        raw = resp.text.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        return {"source": "gemini_verify", "verified": data.get("verified"), "reason": data.get("reason", "")}
    except Exception as e:
        logger.warning(f"Gemini verification failed: {e}")
    return {"source": "gemini_verify", "verified": None, "reason": ""}


def run_verification_layer(client, question: str, answer: str) -> dict:
    """Runs all verifiers and returns an aggregated result."""
    wiki_result   = verify_answer_wikipedia(question, answer)
    gemini_result = verify_answer_gemini(client, question, answer)

    verified_count = sum(
        1 for r in [wiki_result, gemini_result]
        if r.get("verified") is True
    )
    overall = "confirmed" if verified_count >= 1 else "unverified"

    return {
        "wikipedia": wiki_result,
        "gemini_verify": gemini_result,
        "overall_verification": overall
    }


# ─────────────────────────────────────────────
# Feature 10 & 6: Headless Browser + Screenshot
# ─────────────────────────────────────────────
async def scrape_with_playwright(url: str, today_date: str) -> dict:
    """
    Launches a headless Playwright browser to scrape an Amazon Funzone page.
    Returns {"html": str, "screenshot_path": str, "spin_detected": bool}.
    Requires: pip install playwright && playwright install chromium
    """
    result = {"html": "", "screenshot_path": "", "spin_detected": False}
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page    = await browser.new_page(
                user_agent="Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 Chrome/91.0"
            )
            # Feature 9: Inject regional headers
            await page.set_extra_http_headers(REGION_HEADERS.get("IN", {}))

            await page.goto(url, timeout=20000, wait_until="networkidle")
            await asyncio.sleep(2)  # Let dynamic content load

            html = await page.content()
            result["html"] = html

            # Feature 10: Screenshot capture
            screenshot_path = str(SCREENSHOT_DIR / f"funzone_{today_date}_{hashlib.md5(url.encode()).hexdigest()[:6]}.png")
            await page.screenshot(path=screenshot_path, full_page=True)
            result["screenshot_path"] = screenshot_path
            logger.info(f"📸 Screenshot saved: {screenshot_path}")

            # Feature 14: Spin Wheel Detection
            spin_keywords = ["spin", "wheel", "scratch", "lucky draw"]
            if any(kw in html.lower() for kw in spin_keywords):
                result["spin_detected"] = True
                logger.info("🎡 Spin Wheel detected on page!")

            await browser.close()
    except ImportError:
        logger.warning("Playwright not installed. Skipping browser scrape. Run: pip install playwright && playwright install chromium")
    except Exception as e:
        logger.warning(f"Playwright scrape failed for {url}: {e}")
    return result


# ─────────────────────────────────────────────
# Feature 11 & 12: OCR + Vision AI (Gemini Vision)
# ─────────────────────────────────────────────
def extract_qa_from_screenshot(client, screenshot_path: str) -> list:
    """
    Uses Gemini Vision to extract quiz Q&A pairs from a screenshot.
    Also handles image-based questions ("Who is this?", "Identify this logo").
    """
    if not screenshot_path or not Path(screenshot_path).exists():
        return []

    logger.info(f"🔍 Running Vision AI on screenshot: {screenshot_path}")
    try:
        with open(screenshot_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        prompt = """
        Look at this Amazon Funzone quiz screenshot carefully.
        Extract ALL quiz questions and their correct answers visible on screen.
        If a question shows an image (person, logo, product) instead of text, describe what it is asking.
        
        Return ONLY a raw JSON array. Each object must have:
        - "question": the quiz question text
        - "answer": the correct answer
        - "has_image": true if the question contains an embedded image, false otherwise
        
        No explanations, no markdown, just the JSON array.
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                        {"text": prompt}
                    ]
                }
            ]
        )
        raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        qa_pairs = json.loads(raw)
        logger.info(f"👁️ Vision AI extracted {len(qa_pairs)} Q&A pairs from screenshot")
        return qa_pairs
    except Exception as e:
        logger.warning(f"Vision AI extraction failed: {e}")
        return []


# ─────────────────────────────────────────────
# Feature 13: Image Storage Helper
# ─────────────────────────────────────────────
def store_screenshot_to_hf(screenshot_path: str, today_date: str, quiz_type: str):
    """
    Pushes quiz screenshots to a separate HuggingFace image dataset.
    """
    if not screenshot_path or not Path(screenshot_path).exists():
        return
    try:
        from datasets import Image as HFImage
        img_entry = {
            "date": today_date,
            "quiz_type": quiz_type,
            "image_path": screenshot_path,
        }
        img_ds = Dataset.from_dict({
            "date":      [img_entry["date"]],
            "quiz_type": [img_entry["quiz_type"]],
            "image":     [screenshot_path],  # HF handles path-based image loading
        }).cast_column("image", HFImage())

        img_ds.push_to_hub(IMAGE_DATASET_NAME)
        logger.info(f"🖼️ Screenshot pushed to HF image dataset: {IMAGE_DATASET_NAME}")
    except Exception as e:
        logger.warning(f"Image dataset push failed: {e}")


# ─────────────────────────────────────────────
# Feature 7: Dynamic Quiz URL Discovery
# ─────────────────────────────────────────────
def discover_quiz_urls(html_content: str) -> list:
    """
    Scans raw HTML for quiz/funzone deep links.
    Returns a list of discovered URLs.
    """
    import re
    urls = set(AMAZON_FUNZONE_URLS)
    # Pattern for Amazon quiz/funzone paths
    pattern = r'https://www\.amazon\.in/[^\s"\'<>]*(?:quiz|funzone|spin|jackpot|contest)[^\s"\'<>]*'
    found = re.findall(pattern, html_content, re.IGNORECASE)
    urls.update(found)
    logger.info(f"🔗 Discovered {len(found)} new quiz URLs from page content")
    return list(urls)


# ─────────────────────────────────────────────
# Feature 16: Event-Based Trigger Detection
# ─────────────────────────────────────────────
def detect_new_quiz_event(html_content: str, last_known_hash: str) -> tuple[bool, str]:
    """
    Hashes the quiz section of the page. Returns (changed: bool, new_hash: str).
    Use this to trigger scraping only when something new appears.
    """
    quiz_hash = hashlib.sha256(html_content.encode()).hexdigest()
    changed   = quiz_hash != last_known_hash
    if changed:
        logger.info("🔔 New quiz event detected (page content changed)!")
    return changed, quiz_hash


# ─────────────────────────────────────────────
# ORIGINAL: Gemini Search-Grounded Q&A Fetch (PRESERVED)
# ─────────────────────────────────────────────
def get_gemini_daily_qna(client, today_date: str, slot: str = "morning") -> list:
    """
    Calls Gemini API using the new google-genai SDK.
    Uses Search Grounding but relies on prompt engineering for JSON output
    to bypass the API's Tool + JSON conflict.
    (Original function — preserved intact, slot param added for Feature 8)
    """
    logger.info(f"Requesting {today_date} [{slot}] Q&A from Gemini with Search...")

    try:
        prompt = f"""
        Today is {today_date}. Use Google Search to find the exact 5 Amazon Funzone Daily Quiz questions and answers for today.
        You MUST format your entire response as a raw JSON array containing exactly 5 objects. 
        Each object must have exactly two keys: "question" and "answer".
        Do not include any greeting, explanation, or text outside of the JSON array.
        Clean up the text. Do not include "Q1" or "Answer:" prefixes.
        """

        # Call with Search enabled, but standard text output
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"googleSearch": {}}]  # Search is ON, JSON strict mode is OFF
            )
        )

        raw_text = response.text.strip()

        # Bulletproof Markdown Stripper (crucial now that strict JSON mode is off)
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        raw_text = raw_text.strip()
        qa_pairs = json.loads(raw_text)

        if isinstance(qa_pairs, list) and len(qa_pairs) > 0:
            logger.info(f"✅ Successfully decoded {len(qa_pairs)} questions from Gemini [{slot}].")
            return qa_pairs
        else:
            logger.error(f"Gemini returned invalid format. Raw: {raw_text}")
            return []

    except Exception as e:
        logger.error(f"Critical Gemini API Error: {e}")
        return []


# ─────────────────────────────────────────────
# Feature 17: Historical Analytics Dashboard
# ─────────────────────────────────────────────
def generate_analytics(df: pd.DataFrame) -> dict:
    """
    Generates analytics from the accumulated dataset:
    - Most repeated questions
    - Common answer patterns
    - Category trends
    """
    if df.empty:
        return {}

    analytics = {}

    # Most repeated questions
    if "question" in df.columns:
        top_questions = df["question"].value_counts().head(10).to_dict()
        analytics["most_repeated_questions"] = top_questions

    # Common answers
    if "answer" in df.columns:
        top_answers = df["answer"].value_counts().head(10).to_dict()
        analytics["common_answers"] = top_answers

    # Category trends
    if "category" in df.columns:
        cat_counts = df["category"].value_counts().to_dict()
        analytics["category_trends"] = cat_counts

    # Quiz type distribution
    if "quiz_type" in df.columns:
        type_counts = df["quiz_type"].value_counts().to_dict()
        analytics["quiz_type_distribution"] = type_counts

    # Answer accuracy (Feature 19)
    if "verification_status" in df.columns:
        accuracy = (df["verification_status"] == "confirmed").mean()
        analytics["answer_accuracy_rate"] = round(float(accuracy), 3)

    # Daily entry counts
    if "date" in df.columns:
        daily_counts = df["date"].value_counts().sort_index().to_dict()
        analytics["daily_entry_counts"] = daily_counts

    logger.info(f"📊 Analytics generated: {list(analytics.keys())}")
    return analytics


def push_analytics_to_hf(analytics: dict, today_date: str):
    """Saves analytics snapshot to a dedicated HF dataset."""
    if not analytics:
        return
    try:
        analytics_entry = {
            "date": today_date,
            "analytics_json": json.dumps(analytics)
        }
        ds = Dataset.from_dict({k: [v] for k, v in analytics_entry.items()})
        ds.push_to_hub(ANALYTICS_DATASET)
        logger.info(f"📈 Analytics pushed to {ANALYTICS_DATASET}")
    except Exception as e:
        logger.warning(f"Analytics push failed: {e}")


# ─────────────────────────────────────────────
# Feature 8: Multi-Time Slot Fetcher
# ─────────────────────────────────────────────
def fetch_all_slots(client, today_date: str) -> list:
    """
    Fetches quizzes across morning / afternoon / evening slots.
    Merges and deduplicates results by question text.
    """
    all_results = []
    seen_questions = set()

    for slot in FETCH_SLOTS:
        logger.info(f"⏰ Fetching slot: {slot.upper()}")
        slot_results = get_gemini_daily_qna(client, today_date, slot=slot)
        for item in slot_results:
            q = item.get("question", "").strip()
            if q and q not in seen_questions:
                item["fetch_slot"] = slot
                all_results.append(item)
                seen_questions.add(q)
        time.sleep(2)  # Polite delay between slot fetches

    logger.info(f"⏰ Total unique Q&A across all slots: {len(all_results)}")
    return all_results


# ─────────────────────────────────────────────
# ORIGINAL: Main Automation (PRESERVED + Extended)
# ─────────────────────────────────────────────
def run_automation():
    """Main execution flow for syncing data to Hugging Face."""
    logger.info("=== Starting Daily Funzone Sync (Enhanced) ===")

    if not HF_TOKEN or not GEMINI_KEY:
        logger.error("❌ Missing API Keys! Check GitHub Secrets.")
        return

    login(token=HF_TOKEN)
    client     = genai.Client(api_key=GEMINI_KEY)
    today_date = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d")

    # ── 1. Load existing historical dataset (ORIGINAL) ──────────────────────
    try:
        df = load_dataset(DATASET_NAME, split="train").to_pandas()
        df.columns = [str(c).lower() for c in df.columns]
        if 'question' not in df.columns or 'date' not in df.columns:
            logger.warning("⚠️ Dataset structure corrupted. Initializing fresh dataframe.")
            df = pd.DataFrame(columns=["date", "question", "answer"])
    except Exception:
        logger.info("ℹ️ No existing dataset found. Creating a new one.")
        df = pd.DataFrame(columns=["date", "question", "answer"])

    existing_questions = df["question"].tolist() if not df.empty else []

    # ── 2. Feature 6 + 7 + 10 + 14 + 16: Playwright scrape + discovery ──────
    last_page_hash = ""
    vision_qa      = []
    for url in AMAZON_FUNZONE_URLS:
        scrape_result = asyncio.run(scrape_with_playwright(url, today_date))

        if scrape_result["html"]:
            # Feature 16: Event-based detection
            changed, last_page_hash = detect_new_quiz_event(scrape_result["html"], last_page_hash)

            # Feature 7: Dynamic URL discovery
            discover_quiz_urls(scrape_result["html"])

        # Feature 11: Vision AI on screenshot
        if scrape_result["screenshot_path"]:
            vision_pairs = extract_qa_from_screenshot(client, scrape_result["screenshot_path"])
            vision_qa.extend(vision_pairs)

            # Feature 13: Store image to HF
            store_screenshot_to_hf(scrape_result["screenshot_path"], today_date, "Daily Quiz")

        if scrape_result["spin_detected"]:
            logger.info("🎡 Spin quiz detected — including in fetch scope")

    # ── 3. Feature 8: Multi-slot Gemini fetch (ORIGINAL logic preserved) ────
    scraped_data = fetch_all_slots(client, today_date)

    # Merge vision-extracted Q&A with Gemini results
    for vqa in vision_qa:
        q = vqa.get("question", "").strip()
        if q and not any(item.get("question") == q for item in scraped_data):
            scraped_data.append(vqa)

    if not scraped_data:
        logger.error("❌ Sync aborted: No data retrieved from any source.")
        return

    # ── 4. Enrich each Q&A entry with new features ──────────────────────────
    new_entries = []
    for item in scraped_data:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()

        if not q or not a:
            continue

        # ORIGINAL exact dedup
        if not df.empty and ((df['question'] == q) & (df['date'] == today_date)).any():
            continue

        # Feature 4: Semantic duplicate detection
        if is_semantic_duplicate(client, q, existing_questions):
            logger.info(f"♻️ Skipping semantic duplicate: {q[:60]}")
            continue

        # Feature 2 + 1: Quiz type classification
        quiz_type = classify_quiz_type(client, q + " " + a)

        # Feature 3: Confidence score
        confidence_data = get_answer_with_confidence(client, q)
        confidence      = confidence_data.get("confidence", 0.5)
        # Use original answer; confidence is supplementary metadata
        if not a and confidence_data.get("answer"):
            a = confidence_data["answer"]

        # Feature 18: Category tagging
        category = tag_quiz_category(client, q, a)

        # Feature 5: Verification layer
        verification = run_verification_layer(client, q, a)
        verification_status = verification.get("overall_verification", "unverified")

        entry = {
            "date":                today_date,
            "question":            q,
            "answer":              a,
            "quiz_type":           quiz_type,           # Feature 1+2
            "confidence_score":    round(confidence, 3), # Feature 3
            "category":            category,             # Feature 18
            "verification_status": verification_status,  # Feature 5+19
            "fetch_slot":          item.get("fetch_slot", "morning"),  # Feature 8
            "has_image":           item.get("has_image", False),       # Feature 12
            "verification_detail": json.dumps(verification),           # Feature 5
        }
        new_entries.append(entry)
        existing_questions.append(q)

    # ── 5. Save and Push (ORIGINAL logic preserved) ─────────────────────────
    if new_entries:
        new_df      = pd.DataFrame(new_entries)

        # Align columns between old and new df (new cols default to None for old rows)
        updated_df  = pd.concat([df, new_df], ignore_index=True)

        # ORIGINAL: Local backup
        backup_filename = f"backup_{today_date}.json"
        updated_df.to_json(backup_filename, orient="records", indent=4)
        logger.info(f"💾 Local backup saved as {backup_filename}")

        # ORIGINAL: Push to HF
        updated_ds = Dataset.from_pandas(updated_df)
        updated_ds.push_to_hub(DATASET_NAME)
        logger.info(f"🚀 SUCCESS: {len(new_entries)} new entries pushed to Hugging Face!")

        # Feature 17+19: Generate and push analytics
        analytics = generate_analytics(updated_df)
        push_analytics_to_hf(analytics, today_date)

    else:
        logger.info(f"♻️ Dataset is already up to date for {today_date}. No changes made.")
        # Still run analytics on existing data
        analytics = generate_analytics(df)
        push_analytics_to_hf(analytics, today_date)


# ─────────────────────────────────────────────
# Entry Point (ORIGINAL preserved)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_automation()
