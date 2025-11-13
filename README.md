# DATE NECESARE PENTRU PROIECTUL TONX

Acest folder este necesar pentru rularea scripturilor de fine-tuning.

**ATENȚIE:** Fișierele de mari dimensiuni NU sunt stocate în Git.

## 1. Setul de Date Sentiment140 (Polaritate)

- **Sursa:** https://www.kaggle.com/datasets/kazanova/sentiment140
- **Fișier Așteptat:** `training.1600000.processed.noemoticon.csv`
- **Locație a fișierului:** dataset/sentiment140/

## 2. Setul de Date Twitter Neutral

- **Sursa:** https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
- **Fișier Așteptat:** `twitter_training.csv`
- **Locație a fișierului:** dataset/twitter-sentiment-neutral/

## ⚠️ Pasul Obligatoriu

După ce descărcați fișierele, asigurați-vă că le plasați în sub-directoarele specificate (e.g., `data/sentiment140/`) pentru a evita erorile `FileNotFoundError` în scripturile Python.
