import os # Importă DistilBertTokenizerFast, folosit pentru a converti textul în token-uri numerice
from transformers import DistilBertTokenizerFast
from datasets import load_dataset, Value


# DistilBERT este folosit pentru tokenizare.
MODEL_NAME = "distilbert-base-uncased"
# Lungimea maximă a secvenței de token-uri. Textele mai lungi vor fi trunchiate, cele mai scurte umplute (padding).
MAX_LENGTH = 128 
# Directorul de ieșire unde vor fi salvate seturile de date tokenizate.
TOKENIZED_OUTPUT_DIR = os.path.join("data", "tokenized_datasets")

# --- Funcția de Tokenizare ---
def tokenize_function(examples, tokenizer, text_column):
    """ Tokenizează textele folosind coloana specificată. """
    return tokenizer(
        examples[text_column],
        # Trunchiază textul dacă depășește MAX_LENGTH.
        truncation=True,
        # Adaugă padding (umplere) pentru a asigura aceeași lungime.
        padding=True,
        max_length=MAX_LENGTH
    )

# --- Funcția care rulează Tokenizarea pentru un singur Task ---
def run_tokenization_for_task(task: str):
    """
    Încarcă, filtrează și tokenizează seturile de date pentru un task ('sentiment' sau 'category').
    """
    task = task.lower()
    
    # Setează variabilele în funcție de task
    if task == "sentiment":
        prefix = "sentiment"
        text_col = "text" # Coloana de text din setul Sentiment
    elif task == "category":
        prefix = "category"
        text_col = "clean_text" # Coloana de text din setul Categorie
    else:
        print(f"Eroare: Task-ul '{task}' nu este suportat.")
        return None
        
    # --- Căi de Fisiere ---
    # Definește căile către fișierele CSV de intrare
    data_files = {
        "train": os.path.join("data", f"{prefix}_train.csv"),
        "validation": os.path.join("data", f"{prefix}_validation.csv"),
        "test": os.path.join("data", f"{prefix}_test.csv"),
    }
    # Definește calea de salvare a setului de date tokenizat
    output_path = os.path.join(TOKENIZED_OUTPUT_DIR, prefix)
    
    print(f"\n--- Încep Tokenizarea pentru TASK: {task.upper()} ---")
    
    # 1. Încărcare Date
    try:
        # Încarcă seturile de date (train, validation, test) dintr-un fișier CSV
        raw_datasets = load_dataset("csv", data_files=data_files)
    except FileNotFoundError:
        print(f"Eroare: Nu s-au găsit fișierele CSV pentru task-ul {task}.")
        return None

    # 2. Tipizare (Cast)
    features = raw_datasets["train"].features.copy()
    if text_col not in features:
        print(f"Eroare: Coloana '{text_col}' nu a fost găsită în setul de date.")
        return None
        
    # Asigură-te că coloana de text este tratată ca șir de caractere (string)
    features[text_col] = Value('string') 
    raw_datasets = raw_datasets.cast(features)

    # 3. Filtrare Text Invalid
    def filter_invalid_text(example):
        text_value = example.get(text_col)
        # Returnează True dacă valoarea este un string valid și nu este goală (după strip)
        return isinstance(text_value, str) and text_value.strip() != ""

    # Aplică filtrarea pe setul de date, rulând în paralel pe toate nucleele CPU
    raw_datasets = raw_datasets.filter(filter_invalid_text, num_proc=os.cpu_count())
    
    print(f"Dimensiuni după filtrare: Train={len(raw_datasets['train'])}, Validation={len(raw_datasets['validation'])}")
    
    # 4. Tokenizare
    # Încarcă tokenizer-ul pre-antrenat DistilBERT
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Aplică funcția de tokenizare pe setul de date
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_function(examples, tokenizer, text_col),
        batched=True, # Procesează mai multe exemple deodată (mai rapid)
        num_proc=os.cpu_count(),
        # Păstrăm doar 'input_ids', 'attention_mask' (produse de tokenizer) și 'label'
        # Elimină coloana de text originală și alte coloane redundante
        remove_columns=[col for col in raw_datasets["train"].column_names if col not in ['label']],
    )

    # 5. Salvare
    # Setează formatul de ieșire ca tensori PyTorch
    tokenized_datasets.set_format("torch")
    
    print(f"Salvez setul de date tokenizat în: {output_path}")
    # Salvează setul de date tokenizat pe disc
    tokenized_datasets.save_to_disk(output_path)

    print(f"---  Tokenizare {task.upper()} FINALIZATĂ ---")
    return tokenized_datasets


if __name__ == '__main__':
    # Crează directorul de output dacă nu există
    os.makedirs(TOKENIZED_OUTPUT_DIR, exist_ok=True)
    
    # Rulare pentru Sentiment
    run_tokenization_for_task(task="sentiment")
    
    # Rulare pentru Categorie
    run_tokenization_for_task(task="category")
    
    print("\nToate seturile de date au fost tokenizate și salvate!")