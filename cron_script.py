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
    
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={"response_mime_type": "application/json"}
    )
    
    prompt = f"""
    Today is {today_date}. You must find the exact 5 Amazon Funzone Daily Quiz questions and answers for today.
    Return ONLY a JSON array containing exactly 5 objects. 
    Each object must have exactly two keys: "question" and "answer".
    Clean up the text. Do not include "Q1" or "Answer:" prefixes.
    """
    
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # --- NEW: Bulletproof Markdown Stripper ---
        # Sometimes LLMs wrap JSON in ```json ... ``` blocks. This removes them.
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        raw_text = raw_text.strip()
        # ------------------------------------------

        qa_pairs = json.loads(raw_text)
        
        if isinstance(qa_pairs, list) and len(qa_pairs) > 0:
            print(f"Successfully generated {len(qa_pairs)} questions from Gemini.")
            return qa_pairs
        else:
            print(f"Gemini returned data, but it was not a valid list. Raw output: {raw_text}")
            return []
            
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return []
