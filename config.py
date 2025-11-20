"""
Configurări pentru proiectul de analiză a tonului
"""

# Căi fișiere
SENTIMENT140_PATH = "data/neprocesat/training.1600000.processed.noemoticon.csv"
TWITTER_SENTIMENT_PATH = "data/neprocesat/twitter_training.csv"
OUTPUT_DIR = "data/procesat/"

# Coloane necesare
SENTIMENT140_COLUMNS = {
    'target': 'target',  # 0 = negative, 4 = positive
    'text': 'text'
}

TWITTER_SENTIMENT_COLUMNS = {
    'sentiment': 'target',  # pentru neutral
    'text': 'text'
}

# Mapare etichete
LABEL_MAPPING = {
    0: 'negative',
    2: 'neutral', 
    4: 'positive'
}

# Configurări pentru balansare
BALANCE_METHOD = 'undersample'  # sau 'oversample'
TARGET_SAMPLES_PER_CLASS = None  # None = automat (minim dintre clase)

# Configurări pentru procesare text
MIN_TEXT_LENGTH = 5  # caractere minime
MAX_TEXT_LENGTH = 280  # caractere maxime (Twitter limit)

# Random seed pentru reproducibilitate
RANDOM_SEED = 42