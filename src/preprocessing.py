import re
import pandas as pd

# coloanele din fisierul noemoticon.csv (sentiment140)
RAW_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]


def simple_clean(text):
    """
    Functie simpla care curata textul unui tweet.
    """
    if not isinstance(text, str):
        text = str(text)

    # facem totul cu litere mici
    text = text.lower()

    # scoatem link-urile
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # scoatem @user
    text = re.sub(r"@\w+", " ", text)

    # pastram doar litere si spatii
    text = re.sub(r"[^a-z\s]", " ", text)

    # scoatem spatiile multiple
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_kaggle_neutral(path):
    """
    Citim fisierul twitter_training.csv de la Kaggle.
    Fisierul are 4 coloane, dar nu are header, asa ca le punem noi:
    id, entity, sentiment, text

    Din el ne intereseaza doar tweet-urile cu sentiment "Neutral".
    Le intoarcem cu label = 2 (neutru) si text curatat.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["id", "entity", "sentiment", "text"]
    )

    # pastram doar randurile unde sentimentul este Neutral
    df_neutral = df[df["sentiment"].astype(str).str.lower() == "neutral"].copy()

    # curatam textul
    df_neutral["text"] = df_neutral["text"].astype(str).apply(simple_clean)

    # punem label 2 pentru neutru
    df_neutral["label"] = 2

    # intoarcem doar coloanele de care avem nevoie
    return df_neutral[["label", "text"]]


def make_clean_csv():
 
    # cai catre fisiere
    sentiment140_path = "data/noemoticon.csv"
    kaggle_path = "data/twitter_training.csv"

    out_all = "data/trainingdata.csv"
    out_train = "data/train.csv"
    out_val = "data/validation.csv"
    out_test = "data/test.csv"

    # ================== Sentiment140 ==================
    df_s140 = pd.read_csv(
        sentiment140_path,
        encoding="latin-1",
        header=None,
        names=RAW_COLUMNS
    )

    # pastram doar target 0 (negativ) si 4 (pozitiv)
    df_s140 = df_s140[df_s140["target"].isin([0, 4])]

    # mapam in label 0 si 1
    df_s140["label"] = df_s140["target"].map({0: 0, 4: 1})

    # curatam textul
    df_s140["text"] = df_s140["text"].astype(str).apply(simple_clean)

    # pastram doar ce ne intereseaza
    df_s140 = df_s140[["label", "text"]]

    # ================== Kaggle Neutral ==================
    df_neutral = load_kaggle_neutral(kaggle_path)

    # ================== Combinam ==================
    df_all = pd.concat([df_s140, df_neutral], ignore_index=True)

    # salvam tot dataset-ul combinat
    df_all.to_csv(out_all, index=False)
    print("Am salvat toate datele in:", out_all)

    # ================== Split 70 / 15 / 15 ==================
    # amestecam randurile
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df_all)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    # restul o sa fie test

    df_train = df_all.iloc[:n_train]
    df_val = df_all.iloc[n_train:n_train + n_val]
    df_test = df_all.iloc[n_train + n_val:]

    # salvam
    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)
    df_test.to_csv(out_test, index=False)

    print("Train salvat in:", out_train)
    print("Validation salvat in:", out_val)
    print("Test salvat in:", out_test)


