import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset
from huggingface_hub import login
from datetime import datetime
import pytz
import re

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_NAME = "kacapower/funzone-qna" # Update this to your hf dataset path

# --- Scraper Functions ---
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def extract_qa_from_text(text_elements):
    """
    Looks for patterns like Q1... and Answer: ... in a list of text elements.
    This is much more robust than relying on exact CSS classes.
    """
    qa_pairs = []
    current_q = None
    
    for text in text_elements:
        text = text.strip()
        if not text:
            continue
            
        # Detect Question (Usually ends with ? or starts with Q)
        if ("?" in text or text.lower().startswith("q")) and "answer" not in text.lower():
            current_q = re.sub(r'^(Q\d*\.?\s*|\d+\.\s*)', '', text) # Clean up "Q1. " prefix
            
        # Detect Answer
        elif current_q and ("answer" in text.lower() or "ans:" in text.lower() or "ans " in text.lower()):
            # Clean up "Answer: " prefix
            ans = re.sub(r'^(Answer\s*:?\s*|Ans\s*:?\s*)', '', text, flags=re.IGNORECASE)
            qa_pairs.append({"question": current_q, "answer": ans})
            current_q = None # Reset for the next question
            
    return qa_pairs

def scrape_site(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all paragraph, list items, and headings
        elements = [el.get_text(separator=' ') for el in soup.find_all(['p', 'li', 'h3', 'h4', 'div'])]
        
        qa_pairs = extract_qa_from_text(elements)
        return qa_pairs
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return []

def get_daily_qna():
    # Primary Source
    print("Attempting Buyhatke...")
    data = scrape_site("https://buyhatke.com/amazon-daily-quiz-answers-today")
    
    # Fallback Source
    if not data or len(data) < 4: 
        print("Buyhatke failed or returned too few results. Trying Dealsmagnet fallback...")
        data = scrape_site("https://www.dealsmagnet.com/amazon-quiz-answers")
        
    return data

# --- Main Logic ---
def run_automation():
    login(token=HF_TOKEN)
    
    # 1. Load existing database safely
    try:
        df = load_dataset(DATASET_NAME, split="train").to_pandas()
        # Force column names to lowercase to prevent KeyErrors (e.g., "Question" -> "question")
        df.columns = [str(c).lower() for c in df.columns]
        
        # If the dataset is somehow corrupted or missing core columns, reset it
        if 'question' not in df.columns or 'date' not in df.columns:
            print("Warning: Dataset columns mismatched. Resetting dataframe structure.")
            df = pd.DataFrame(columns=["date", "question", "answer"])
    except Exception:
        df = pd.DataFrame(columns=["date", "question", "answer"])

    # 2. Scrape Today's Q&A
    today_date = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d")
    scraped_data = get_daily_qna()

    if not scraped_data:
        print("Error: Both scrapers failed to find data today.")
        return

    # 3. Deduplicate and Prepare New Entries
    new_entries = []
    for item in scraped_data:
        q = item["question"]
        a = item["answer"]
        # Only add if the question isn't already saved today
        if df.empty or not ((df['question'] == q) & (df['date'] == today_date)).any():
            new_entries.append({"date": today_date, "question": q, "answer": a})

    # 4. Push to hf Dataset
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_ds = Dataset.from_pandas(updated_df)
        updated_ds.push_to_hub(DATASET_NAME)
        print(f"Successfully pushed {len(new_entries)} new questions to hf dataset.")
    else:
        print("No new questions found. Dataset is already up to date.")

if __name__ == "__main__":
    run_automation()
