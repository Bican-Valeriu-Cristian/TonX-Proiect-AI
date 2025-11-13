import re
import pandas as pd

RAW_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]


def simple_clean(text: str) -> str:
    """Curăță foarte simplu textul unui tweet."""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()                          # litere mici
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # scoate link-uri
    text = re.sub(r"@\w+", " ", text)              # scoate @user
    text = re.sub(r"[^a-z\s]", " ", text)          # scoate tot ce nu e litera/spațiu
    text = re.sub(r"\s+", " ", text).strip()       # spații în plus

    return text


def make_clean_csv(test_size: float = 0.2, random_state: int = 42):
    """
    Curăță datele brute și le salvează în:
      - data/trainingdata.csv  (tot setul curățat)
      - data/train.csv         (date de antrenare)
      - data/test.csv          (date de testare)
    """

    input_path = "data/noemoticon.csv"
    full_output_path = "data/trainingdata.csv"
    train_output_path = "data/train.csv"
    test_output_path = "data/test.csv"

    # citim datele brute
    df = pd.read_csv(
        input_path,
        encoding="latin-1",
        header=None,
        names=RAW_COLUMNS,
    )

    # 0 = negativ, 4 = pozitiv -> normalizam datele
    df["label"] = df["target"].map({0: 0, 4: 1})

    # curățăm text
    df["text"] = df["text"].astype(str).apply(simple_clean)

    # păstrăm doar coloanele relevante
    df_clean = df[["label", "text"]]

    # salvăm tot setul curățat
    df_clean.to_csv(full_output_path, index=False)
    print(f"Am salvat datasetul curățat complet în: {full_output_path}")

    # facem split train / test
    train_frac = 1 - test_size
    df_train = df_clean.sample(frac=train_frac, random_state=random_state)
    df_test = df_clean.drop(df_train.index)

    # pentru siguranță, reindexăm
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # salvăm fișierele train / test
    df_train.to_csv(train_output_path, index=False)
    df_test.to_csv(test_output_path, index=False)

    print(f"Am salvat datele de antrenare în: {train_output_path}")
    print(f"Am salvat datele de testare în: {test_output_path}")


if __name__ == "__main__":
    make_clean_csv()
