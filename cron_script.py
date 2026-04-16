import os
import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import login
import google.generativeai as genai
from datetime import datetime
import pytz
import json

# --- Config ---
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
DATASET_NAME = "kacapower/funzone-qna"

def get_gemini_daily_qna(today_date):
    """Calls Gemini API and forces it to return a clean JSON array."""
    print(f"Calling Gemini API for {today_date} questions...")
    
    genai.configure(api_key=GEMINI_KEY)
    
    # Using Gemini 1.5 Flash. We enforce JSON output so the script doesn't break parsing text.
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={"response_mime_type": "application/json"}
    )
    
    # The prompt explicitly anchors the LLM to today's date
    prompt = f"""
    Today is {today_date}. You must find the exact 5 Amazon Funzone Daily Quiz questions and answers for today.
    Return ONLY a JSON array containing exactly 5 objects. 
    Each object must have exactly two keys: "question" and "answer".
    Clean up the text. Do not include "Q1" or "Answer:" prefixes.
    """
    
    try:
        response = model.generate_content(prompt)
        # Parse the JSON string returned by Gemini into a Python list of dictionaries
        qa_pairs = json.loads(response.text)
        
        # Verify we got a list back
        if isinstance(qa_pairs, list) and len(qa_pairs) > 0:
            return qa_pairs
        else:
            print("Gemini returned JSON, but it was not a valid list.")
            return []
            
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return []

def run_automation():
    login(token=HF_TOKEN)
    
    # 1. Load existing database safely
    try:
        df = load_dataset(DATASET_NAME, split="train").to_pandas()
        df.columns = [str(c).lower() for c in df.columns]
        if 'question' not in df.columns or 'date' not in df.columns:
            df = pd.DataFrame(columns=["date", "question", "answer"])
    except Exception:
        df = pd.DataFrame(columns=["date", "question", "answer"])

    # 2. Setup Today's Date
    today_date = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d")
    
    # 3. Fetch from Gemini
    scraped_data = get_gemini_daily_qna(today_date)

    if not scraped_data:
        print("Error: Failed to retrieve valid Q&A from Gemini API today.")
        return

    # 4. Deduplicate and Prepare New Entries
    new_entries = []
    for item in scraped_data:
        # Failsafe in case Gemini hallucinates keys
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        
        if not q or not a:
            continue
            
        # Only add if the exact question isn't already saved today
        if df.empty or not ((df['question'] == q) & (df['date'] == today_date)).any():
            new_entries.append({"date": today_date, "question": q, "answer": a})

    # 5. Push to HF Dataset
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_ds = Dataset.from_pandas(updated_df)
        updated_ds.push_to_hub(DATASET_NAME)
        print(f"Successfully pushed {len(new_entries)} LLM-generated questions to hf dataset.")
    else:
        print("No new questions found. Dataset is already up to date.")

if __name__ == "__main__":
    run_automation()
