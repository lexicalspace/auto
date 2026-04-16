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
DATASET_NAME = "kacapower/funzone-qna" 

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def extract_strict_qa(text_elements):
    """
    Strictly extracts exactly 5 questions.
    Cleans up prefixes like "Question 1 of 5:" and "The answer is-".
    """
    qa_pairs = []
    current_q = None
    
    for text in text_elements:
        text = text.strip()
        if not text:
            continue
            
        # 1. Identify Questions: Must start with Q1, Question 1, etc., and contain a question mark
        if re.match(r'^(Q(?:uestion)?\s*\d+)', text, re.IGNORECASE) and "?" in text:
            # Strip the prefix (removes "Q1.", "Question 1:", "Question 1 of 5:")
            clean_q = re.sub(r'^(Q(?:uestion)?\s*\d+(?:\s*of\s*\d+)?\s*[:.)-]?\s*)', '', text, flags=re.IGNORECASE)
            current_q = clean_q.strip()
            
        # 2. Identify Answers: Must immediately follow a question and start with Ans, Answer, or The answer
        elif current_q and re.match(r'^(Ans|The answer|Correct Answer)', text, re.IGNORECASE):
            # Strip the prefix (removes "Answer:", "The answer is-", "Correct Answer- ✅")
            clean_a = re.sub(r'^(Answer\s*[:.-]?\s*|Ans\s*[:.-]?\s*|The answer is\s*[-:]?\s*|Correct Answer\s*[-:]?\s*✅?\s*)', '', text, flags=re.IGNORECASE)
            clean_a = clean_a.replace('✅', '').strip()
            
            # Avoid duplicate captures
            if not any(item['question'] == current_q for item in qa_pairs):
                qa_pairs.append({"question": current_q, "answer": clean_a})
                
            current_q = None # Reset for the next question
            
    # Return only the first 5 legitimate pairs (standard Amazon Quiz size)
    return qa_pairs[:5]

def scrape_site(url):
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Grab all text elements that might contain the Q&A
        elements = [el.get_text(separator=' ', strip=True) for el in soup.find_all(['p', 'li', 'h3', 'div'])]
        
        qa_pairs = extract_strict_qa(elements)
        return qa_pairs
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return []

def get_daily_qna():
    # Primary Source: Smartprix (Very reliable daily updates)
    print("Attempting Smartprix...")
    data = scrape_site("https://www.smartprix.com/bytes/amazon-daily-quiz-answers/")
    
    # Fallback Source: Dealnloot
    if not data or len(data) < 5: 
        print("Smartprix failed or incomplete. Trying Dealnloot fallback...")
        data = scrape_site("https://www.dealnloot.com/amazon-quiz-answers/")
        
    return data

# --- Main Logic ---
def run_automation():
    login(token=HF_TOKEN)
    
    # Load existing database safely
    try:
        df = load_dataset(DATASET_NAME, split="train").to_pandas()
        df.columns = [str(c).lower() for c in df.columns]
        if 'question' not in df.columns or 'date' not in df.columns:
            df = pd.DataFrame(columns=["date", "question", "answer"])
    except Exception:
        df = pd.DataFrame(columns=["date", "question", "answer"])

    # Scrape Today's Q&A
    today_date = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d")
    scraped_data = get_daily_qna()

    if not scraped_data:
        print("Error: Both scrapers failed to find data today.")
        return

    # Deduplicate and Prepare New Entries
    new_entries = []
    for item in scraped_data:
        q = item["question"]
        a = item["answer"]
        # Only add if the exact question isn't already saved today
        if df.empty or not ((df['question'] == q) & (df['date'] == today_date)).any():
            new_entries.append({"date": today_date, "question": q, "answer": a})

    # Push to hf Dataset
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_ds = Dataset.from_pandas(updated_df)
        updated_ds.push_to_hub(DATASET_NAME)
        print(f"Successfully pushed {len(new_entries)} clean questions to hf dataset.")
    else:
        print("No new questions found. Dataset is already up to date.")

if __name__ == "__main__":
    run_automation()
