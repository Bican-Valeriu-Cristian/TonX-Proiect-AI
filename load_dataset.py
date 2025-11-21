import pandas as pd # type: ignore[import]
import numpy as np # type: ignore[import]
from sklearn.model_selection import train_test_split # type: ignore[import]
import re
import config 
 
#Încărcare Sentiment140 
print(" Încărcare Sentiment140...")
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
ds_sentiment140 = pd.read_csv(config.SENTIMENT140_PATH, 
                       encoding='latin-1', 
                       names=columns)
ds_sentiment140 = ds_sentiment140[['target', 'text']].copy()

# Mapare: 0->0 (negativ), 4->2 (pozitiv)
ds_sentiment140['target'] = ds_sentiment140['target'].map({0: 0, 4: 2})

print(f"Sentiment140: {ds_sentiment140.shape}")
print(f"Distribuție: {ds_sentiment140['target'].value_counts().to_dict()}")


#  Încărcare dataset cu neutre 
print("\n Încărcare dataset cu neutre...")

try:
    df_neutral = pd.read_csv(config.TWITTER_SENTIMENT_PATH, header=None)
    # Format: [id, entity, sentiment, text]
    df_neutral.columns = ['id', 'entity', 'sentiment', 'text']
    
    # Mapare sentimente: Positive->2, Negative->0, Neutral->1, Irrelevant->drop
    sentiment_map = {
        'Positive': 2,
        'Negative': 0,
        'Neutral': 1,
        'Irrelevant': 1
    }
    df_neutral['target'] = df_neutral['sentiment'].map(sentiment_map)
    df_neutral = df_neutral[['target', 'text']]
    
    print(f"Dataset cu neutre: {df_neutral.shape}")
    print(f"Distribuție: {df_neutral['target'].value_counts().to_dict()}")
    
except FileNotFoundError:
    print("⚠️ Fișierul 'twitter_training.csv' nu a fost găsit!")
    print("Descarcă-l de la: kaggle datasets download -d jp797498e/twitter-entity-sentiment-analysis")
    print("\nCREEZ UN DATASET DEMO pentru exemplificare...")
    
   
