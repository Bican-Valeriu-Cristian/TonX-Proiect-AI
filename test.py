import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from datasets import load_from_disk
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importăm arhitectura modelului
from src.model import TaskClassifier

# --- CONFIGURĂRI ---
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_testing(task):
    print(f"\n=======================================================")
    print(f" PORNIRE TESTARE | TASK: {task.upper()}")
    print(f"=======================================================")

    # 1. Încărcare Date Tokenizate
    data_path = os.path.join("data", "tokenized_datasets", task)
    if not os.path.exists(data_path):
        print(f"EROARE: Nu găsesc datele în '{data_path}'.")
        return

    dataset_dict = load_from_disk(data_path)
    
    if 'test' not in dataset_dict:
        print("EROARE: Setul de date nu conține split-ul 'test'.")
        return
    
    test_data = dataset_dict['test']
    
    # Calculăm numărul de clase ÎNAINTE de conversia PyTorch ---
    print(" INFO: Analiză clase...")
    all_labels = np.array(test_data['label']) 
    unique_classes = np.unique(all_labels)
    num_classes = len(unique_classes)
    print(f" INFO: S-au detectat {num_classes} clase în setul de test.")
    # -------------------------------------------------------------------------

    # 2. Setare format PyTorch
    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # 3. Încărcare Model
    model = TaskClassifier(num_classes=num_classes)
    
    model_path = os.path.join("models", f"{task}_best_model.bin")
    
    if not os.path.exists(model_path):
        print(f"EROARE: Nu găsesc modelul salvat la '{model_path}'. Rulează 'train.py' întâi!")
        return
        
    print(f" INFO: Încărcare model din {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 4. Bucla de Predicție
    print(" INFO: Generare predicții...")
    
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 5. Raportare Rezultate
    acc = accuracy_score(true_labels, predictions)
    print(f"\nREZULTATE FINALE PENTRU {task.upper()}:")
    print(f"Acuratețe (Accuracy): {acc:.4f}")
    print("-" * 30)
    
    if task == 'sentiment':
        target_names = ['Negativ', 'Pozitiv'] if num_classes == 2 else None
    else:
        target_names = None

    print(classification_report(true_labels, predictions, target_names=target_names, digits=4))
    
    # 6. Generare Grafic Seaborn (Matricea de Confuzie)
    print("-" * 30)
    print("Generare grafic Matrice de Confuzie...")
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicție')
    plt.ylabel('Realitate (Adevăr)')
    plt.title(f'Matrice de Confuzie - {task.capitalize()}')
    
    # Salvăm imaginea în folderul curent
    img_name = f"confusion_matrix_{task}.png"
    plt.savefig(img_name)
    print(f" >> Graficul a fost salvat ca imagine: {img_name}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="sentiment sau category")
    args = parser.parse_args()
    
    if args.task not in ['sentiment', 'category']:
        print("Eroare: Task-ul trebuie să fie 'sentiment' sau 'category'.")
    else:
        run_testing(args.task)