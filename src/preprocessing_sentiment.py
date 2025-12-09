import re 
import pandas as pd 
from .preprocessing import simple_clean # Importă funcția locală de curățare a textului.

# coloanele din fisierul noemoticon.csv (sentiment140)
RAW_COLUMNS = ["target", "ids", "date", "flag", "user", "text"] # Definește numele coloanelor brute din Sentiment140.


def load_kaggle_neutral(path):
 """
 Citeste fisierul twitter_training.csv.
 Din el ne intereseaza doar tweet-urile cu sentiment "Neutral".
 Le intoarcem cu label = 2 (neutru) si text curatat.
 """
 # Citeste CSV-ul Kaggle, fără header, definind numele coloanelor.
 df = pd.read_csv(
  path,
  header=None,
  names=["id", "entity", "sentiment", "text"]
 )

 # Filtrează DataFrame-ul, păstrând doar rândurile cu sentimentul "neutral".
 df_neutral = df[df["sentiment"].astype(str).str.lower() == "neutral"].copy()

 # Curăță coloana de text folosind funcția importată.
 df_neutral["text"] = df_neutral["text"].astype(str).apply(simple_clean)

 # Adaugă coloana 'label' cu valoarea 2 (convenție pentru Neutral).
 df_neutral["label"] = 2

 # Returnează doar coloanele necesare ('label' și 'text').
 return df_neutral[["label", "text"]]


def make_clean_csv():
 # Căi către fișierele de intrare.
 sentiment140_path = "data/noemoticon.csv"
 kaggle_path = "data/twitter_training.csv"

 out_all = "data/sentiment_alldata.csv"  # Calea de salvare a setului combinat.
 out_train = "data/sentiment_train.csv" 
 out_val = "data/sentiment_validation.csv" 
 out_test = "data/sentiment_test.csv" 

 # ================== Sentiment140 ==================
 print("Preprocesare Sentiment140...")
 # Citeste Sentiment140 CSV, cu encoding specific (latin-1 „șțăî„)
 df_s140 = pd.read_csv(
  sentiment140_path,
  encoding="latin-1",
  header=None,
  names=RAW_COLUMNS
 )

 # Filtrează, păstrând doar sentimentele 0 (Negativ) și 4 (Pozitiv).
 df_s140 = df_s140[df_s140["target"].isin([0, 4])]

 # Mapează etichetele Sentiment140 (0 și 4) la noile etichete (0 și 1).
 df_s140["label"] = df_s140["target"].map({0: 0, 4: 1})

 # Curăță textul din Sentiment140.
 df_s140["text"] = df_s140["text"].astype(str).apply(simple_clean)

 # Păstrează doar coloanele finale ('label' și 'text').
 df_s140 = df_s140[["label", "text"]]

 # ================== Kaggle Neutral ==================
 print("Incarcare Kaggle Neutral...")
 # Apelez funcția definită mai sus pentru a obține datele Neutre.
 df_neutral = load_kaggle_neutral(kaggle_path)

 # ================== Combinam ==================
 # Unesc datele Sentiment140 (Negativ/Pozitiv) cu cele Neutre.
 df_all = pd.concat([df_s140, df_neutral], ignore_index=True)

 # Amestecă rândurile aleatoriu (frac=1) înainte de split-uire (seed=42 pentru reproductibilitate).
 df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

 # Salvează DataFrame-ul combinat și amestecat.
 df_all.to_csv(out_all, index=False)
 print("Am salvat toate datele in:", out_all)

 # ================== Split 70 / 15 / 15 ==================
 print("Aplicare Split 70/15/15...")
 n = len(df_all) # Numărul total de mostre.
 n_train = int(0.70 * n) # Calculează numărul de mostre pentru Train (70%).
 n_val = int(0.15 * n) # Calculează numărul de mostre pentru Validation (15%).
 # Restul (15%) va fi automat setul de testare.

 # Selectează primele rânduri pentru setul de antrenare.
 df_train = df_all.iloc[:n_train]
 # Selectează rândurile următoare pentru setul de validare.
 df_val = df_all.iloc[n_train:n_train + n_val]
 # Selectează rândurile rămase pentru setul de testare.
 df_test = df_all.iloc[n_train + n_val:]

 # Salvează setul de antrenare ca CSV.
 df_train.to_csv(out_train, index=False)
 # Salvează setul de validare ca CSV.
 df_val.to_csv(out_val, index=False)
 # Salvează setul de testare ca CSV.
 df_test.to_csv(out_test, index=False)

 print("Train Sentiment salvat in:", out_train)
 print("Validation Sentiment salvat in:", out_val)
 print("Test Sentiment salvat in:", out_test)


if __name__ == "__main__":
 make_clean_csv()