import re

def simple_clean(text):
    """
    Functie simpla care curata textul unui tweet/text.
    """
    if not isinstance(text, str):
        text = str(text)

    # litere mici
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