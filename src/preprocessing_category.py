from datasets import load_dataset, DatasetDict 
import pandas as pd 
import os

# Import local: Funcția de curățare a textului din modulul 'preprocessing'.
from .preprocessing import simple_clean 

LOCAL_FILE_PATH = "data/full_dataset.csv" # Definește calea către fișierul sursă.

def prepare_local_splits():
 
 # Căile de salvare pentru split-uri.
 out_train = "data/category_train.csv"
 out_val = "data/category_validation.csv"
 out_test = "data/category_test.csv"

 # Verifică dacă fișierul de intrare există; dacă nu, afișează eroare și oprește execuția.
 if not os.path.exists(LOCAL_FILE_PATH):
  print(f"EROARE: Nu găsesc fișierul '{LOCAL_FILE_PATH}'.")
  return
 print(f"Incarcarea dataset-ului local din: {LOCAL_FILE_PATH}...")
 
 # Încarcă fișierul CSV în format Hugging Face Dataset.
 dataset = load_dataset('csv', data_files=LOCAL_FILE_PATH)
 full_dataset = dataset['train'] # Extrage setul complet de date (încărcat inițial în 'train').

 def process_data(examples):  # Funcția care aplică transformările pe fiecare rând/exemplu.
  examples["clean_text"] = [simple_clean(str(t)) for t in examples["text"]] # Aplică curățarea textului și creează coloana 'clean_text'.
  examples["label"] = examples["category_id"] # Redenumește 'category_id' în 'label'.
  return examples

 # Identifică coloanele originale de eliminat.
 cols_in_dataset = full_dataset.column_names
 cols_to_remove = [c for c in cols_in_dataset if c not in ["label", "clean_text"]]

 print("Procesare text...")
 # Aplică funcția de procesare pe întregul set.
 processed_dataset = full_dataset.map(
  process_data, 
  remove_columns=cols_to_remove, # Elimină coloanele inutile.
  batched=True # Procesează în loturi pentru viteză.
 )
 
 print("Realizare split 70% Train - 15% Val - 15% Test...")

 # Split inițial: 70% Train și 30% Temp (Temp = Val + Test).
 train_test_split = processed_dataset.train_test_split(test_size=0.3, seed=42)
 dataset_train = train_test_split['train'] 
 dataset_temp = train_test_split['test'] 

 # Split secundar: Împarte Temp (30%) în 15% Val și 15% Test.
 val_test_split = dataset_temp.train_test_split(test_size=0.5, seed=42)
 dataset_val = val_test_split['train'] 
 dataset_test = val_test_split['test'] 

 # Creează un DatasetDict care conține cele trei split-uri.
 final_splits = DatasetDict({
  "train": dataset_train,
  "validation": dataset_val,
  "test": dataset_test
 })
 
 # Coloanele dorite în fișierul final.
 COLOANE_FINALE = ['label', 'clean_text'] 

 print("Salvare fișiere CSV...")
 # Salvează split-ul Train ca CSV (conversie la Pandas).
 final_splits["train"].to_pandas()[COLOANE_FINALE].to_csv(out_train, index=False)
 # Salvează split-ul Validation ca CSV.
 final_splits["validation"].to_pandas()[COLOANE_FINALE].to_csv(out_val, index=False)
 # Salvează split-ul Test ca CSV.
 final_splits["test"].to_pandas()[COLOANE_FINALE].to_csv(out_test, index=False)
 
 # Afișează un sumar al rezultatelor.
 print("-" * 30)
 print(f"Train ({len(dataset_train)} mostre) salvat in: {out_train}")
 print(f"Validation ({len(dataset_val)} mostre) salvat in: {out_val}")
 print(f"Test ({len(dataset_test)} mostre) salvat in: {out_test}")
 print("-" * 30)

if __name__ == "__main__":
 prepare_local_splits()