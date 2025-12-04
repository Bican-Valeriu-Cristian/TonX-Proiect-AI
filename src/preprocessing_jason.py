from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd
import re 
import numpy as np 
from preprocessing import simple_clean

JASON_DATASET_NAME = "jason23322/high-accuracy-email-classifier"

def prepare_jason_splits():
    
    # Cai de salvare pentru setul tau final de Categorie
    out_train_cat = "data/category_train.csv"
    out_val_cat = "data/category_validation.csv"
    out_test_cat = "data/category_test.csv"

    print("Incarcarea dataset-ului Jason23322 (Categoria)...")
    
    # Incarcam setul de date. Acesta vine deja impartit in 'train' si 'test'
    hg_dataset_dict = load_dataset(JASON_DATASET_NAME)

    # Functia de mapare: curata textul si selecteaza eticheta corecta
    def process_jason_data(examples):
        # 1. Aplicam curatarea textului - functia din preprocessing.py
        examples["clean_text"] = [simple_clean(t) for t in examples["text"]]
        
        # 2. Renumim 'category_id' in 'label' (0-5) pentru consistenta cu celalalt set de date
        examples["label"] = examples["category_id"]
        
        return examples

    # Aplicam procesarea pe ambele split-uri
    processed_datasets = hg_dataset_dict.map(
        process_jason_data, 
        remove_columns=["id", "subject", "body", "text", "category", "category_id"], # Eliminam coloanele inutile
        batched=True
    )
    
    # Setul original Jason nu avea 'validation'. Il vom crea din setul 'test' (50/50).
    
    temp_split = processed_datasets["test"].train_test_split(test_size=0.5, seed=42)
    
    final_jason_splits = DatasetDict({
        "train": processed_datasets["train"],
        "validation": temp_split["train"],
        "test": temp_split["test"]
    })

    print("Impartire in Train/Validation/Test realizata.")
    
    COLOANE_FINALE = ['label', 'clean_text']

    #Salvarea in format csv    
    df_train = final_jason_splits["train"].to_pandas()
    df_train[COLOANE_FINALE].to_csv(out_train_cat, index=False)

    df_val = final_jason_splits["validation"].to_pandas()
    df_val[COLOANE_FINALE].to_csv(out_val_cat, index=False)
    
    df_test = final_jason_splits["test"].to_pandas()
    df_test[COLOANE_FINALE].to_csv(out_test_cat, index=False)
    
    #afisare numar labels
    print("-" * 30)
    print(f"Pregatire Jason23322 finalizata. Seturi salvate:")
    print(f"Train Categorie: {len(final_jason_splits['train'])} mesaje")
    print(f"Validation Categorie: {len(final_jason_splits['validation'])} mesaje")
    print(f"Test Categorie: {len(final_jason_splits['test'])} mesaje")


# Exemplu de Rulare:
if __name__ == "__main__":
    prepare_jason_splits()
