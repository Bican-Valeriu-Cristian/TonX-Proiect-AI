import load_dataset
import config  # type: ignore
import pandas as pd  # type: ignore[import]
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore[import]
import re 

ds_sentiment140 = load_dataset.ds_sentiment140
df_neutral = load_dataset.df_neutral

def clean_text(text):
    """Curăță un text individual"""
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@username)TonX-Proiect-AI
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (păstrează textul)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove HTML entities
    text = re.sub(r'&\w+;', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_dataset(df, dataset_name):
    """Curăță un dataset complet"""
    print(f"\n🧹 Curățare {dataset_name}...")
    print(f"   Rânduri înainte: {len(df)}")
    
    original_count = len(df)
    
    # 1. Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f"   Duplicate eliminate: {before_dup - len(df)}")
    
    # 2. Remove missing values
    df = df.dropna(subset=['text', 'target'])
    
    # 3. Clean text
    df['text'] = df['text'].apply(clean_text)
    
    # 4. Remove empty or too short texts
    df = df[df['text'].str.len() >= config.MIN_TEXT_LENGTH]
    print(f"   Texte prea scurte eliminate: {original_count - len(df)}")
    
    # 5. Remove too long texts
    df = df[df['text'].str.len() <= config.MAX_TEXT_LENGTH]
    
    # 6. Reset index
    df = df.reset_index(drop=True)
    
    print(f"Rânduri după curățare: {len(df)}")
    print(f"Distribuție: {df['target'].value_counts().to_dict()}")
    
    return df

# Curățare Sentiment140
ds_sentiment140_clean = clean_dataset(ds_sentiment140, "Sentiment140")

# Curățare dataset neutral
if len(df_neutral) > 0:
    df_neutral_clean = clean_dataset(df_neutral, "Twitter Sentiment")
else:
    df_neutral_clean = df_neutral