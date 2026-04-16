"""
Amazon Funzone Daily Fetcher
Backend Automation Engine powered by Gemini Flash (New SDK)
"""

import os
import json
import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import login
from datetime import datetime
import pytz

# --- NEW SDK IMPORTS ---
from google import genai
from google.genai import types

# --- Environment Setup ---
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
DATASET_NAME = "kacapower/funzone-qna" 

def get_gemini_daily_qna(today_date):
    """
    Calls Gemini API using the new google-genai SDK.
    Uses Search Grounding but relies on prompt engineering for JSON output 
    to bypass the API's Tool + JSON conflict.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Requesting {today_date} Q&A from Gemini with Search...")
    
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        
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
                tools=[{"googleSearch": {}}] # Search is ON, JSON strict mode is OFF
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
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Successfully decoded {len(qa_pairs)} questions.")
            return qa_pairs
        else:
            print(f"❌ Error: Gemini returned invalid format. Raw: {raw_text}")
            return []
            
    except Exception as e:
        print(f"❌ Critical Gemini API Error: {e}")
        return []

def run_automation():
    """Main execution flow for syncing data to Hugging Face."""
    print("=== Starting Daily Funzone Sync ===")
    
    if not HF_TOKEN or not GEMINI_KEY:
        print("❌ Missing API Keys! Check GitHub Secrets.")
        return

    login(token=HF_TOKEN)
    today_date = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d")
    
    # 1. Load the existing historical dataset
    try:
        df = load_dataset(DATASET_NAME, split="train").to_pandas()
        df.columns = [str(c).lower() for c in df.columns]
        if 'question' not in df.columns or 'date' not in df.columns:
            print("⚠️ Dataset structure corrupted. Initializing fresh dataframe.")
            df = pd.DataFrame(columns=["date", "question", "answer"])
    except Exception:
        print("ℹ️ No existing dataset found. Creating a new one.")
        df = pd.DataFrame(columns=["date", "question", "answer"])

    # 2. Fetch today's answers
    scraped_data = get_gemini_daily_qna(today_date)
    if not scraped_data:
        print("❌ Sync aborted: No data retrieved.")
        return

    # 3. Deduplicate
    new_entries = []
    for item in scraped_data:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        
        if not q or not a:
            continue
            
        if df.empty or not ((df['question'] == q) & (df['date'] == today_date)).any():
            new_entries.append({"date": today_date, "question": q, "answer": a})

    # 4. Save and Push
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        
        backup_filename = f"backup_{today_date}.json"
        updated_df.to_json(backup_filename, orient="records", indent=4)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Local backup saved as {backup_filename}")
        
        updated_ds = Dataset.from_pandas(updated_df)
        updated_ds.push_to_hub(DATASET_NAME)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 SUCCESS: {len(new_entries)} new entries pushed to Hugging Face!")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ♻️ Dataset is already up to date for {today_date}. No changes made.")

if __name__ == "__main__":
    run_automation()
