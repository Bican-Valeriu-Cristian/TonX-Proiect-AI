import os
from transformers import DistilBertTokenizerFast
from datasets import load_dataset, Value, DatasetDict

# --- Configurație ---
TRAIN_PATH = os.path.join("data", "train.csv")
VAL_PATH = os.path.join("data", "validation.csv")
TEST_PATH = os.path.join("data", "test.csv")
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512 # Lungimea maximă standard pentru DistilBERT

# --- Funcția de Tokenizare ---
def tokenize_function(examples, tokenizer):
    """
    Tokenizează textele din lot.
    """
    # Folosește coloana 'text' și asigură-te că padding-ul și trunchierea sunt aplicate
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

# --- Funcția Principală de Preprocesare ---
def tokenize_data():
    """
    Încarcă, tipizează (cast), filtrează și tokenizează seturile de date.
    """
    print("\n--- 2. Tokenizarea textului (DistilBERT) ---")
    
    # 1. Încarcă datele
    print("Încărc datele pentru tokenizare...")
    data_files = {
        "train": TRAIN_PATH,
        "validation": VAL_PATH,
        "test": TEST_PATH,
    }

    try:
        raw_datasets = load_dataset("csv", data_files=data_files)
    except FileNotFoundError:
        print(f"Eroare: Nu s-au găsit fișierele CSV la calea specificată.")
        return None

    # 2. Asigură tipul de date 'string' pentru coloana 'text'
    print("Asigur tipul de date 'string' pentru coloana 'text'...")
    try:
        features = raw_datasets["train"].features.copy()
        features["text"] = Value('string')
        raw_datasets = raw_datasets.cast(features)
    except KeyError as e:
        print(f"Eroare: Lipsesc coloanele necesare ('text' sau 'label'). Detalii: {e}")
        return None

    # 3. FILTRARE: Elimină rândurile cu valori invalide (None/NaN) din coloana 'text'

    print("Filtrez rândurile cu valori nule sau non-string în coloana 'text'...")
    def filter_invalid_text(example):
        # Verifică dacă valoarea este string ȘI nu este doar spațiu gol
        return isinstance(example["text"], str) and example["text"].strip() != ""

    # Aplicăm filtrarea pe toate split-urile în paralel
    raw_datasets = raw_datasets.filter(
        filter_invalid_text, 
        num_proc=os.cpu_count(),
       
    )
    
    print(f"După filtrare: Train size = {len(raw_datasets['train'])}")
    print(f"După filtrare: Validation size = {len(raw_datasets['validation'])}")
    
    # 4. Încărcare Tokenizator
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # 5. Tokenizează datele
    print(f"Tokenizez datele cu Multi-Processing (num_proc={os.cpu_count()})...")
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        num_proc=os.cpu_count(), # Folosește toate nucleele disponibile
        remove_columns=raw_datasets["train"].column_names # Elimină coloanele originale
    )

    # 6. Formatează pentru PyTorch
    print("Formatare pentru PyTorch...")
    tokenized_datasets.set_format("torch")
    
    # 7. Salvează setul de date tokenizat
    output_dir = os.path.join("data", "tokenized_datasets")
    print(f"Salvez setul de date tokenizat în: {output_dir}")
    tokenized_datasets.save_to_disk(output_dir)

    print("Preprocesare și tokenizare finalizate cu succes!")
    return tokenized_datasets

if __name__ == '__main__':
    pass