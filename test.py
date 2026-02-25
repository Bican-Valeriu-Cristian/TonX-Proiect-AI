import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


# Importăm arhitectura modelului
from src.model import TaskClassifier
from src.metrics_logger import MetricsLogger

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
    
    # Calculăm numărul de clase ÎNAINTE de a seta formatul PyTorch ---
    print(" INFO: Analiză clase...")
    # Extragem etichetele ca o listă simplă sau array numpy
    all_labels = np.array(test_data['label']) 
    unique_classes = np.unique(all_labels)
    num_classes = len(unique_classes)
    
    print(f" INFO: S-au detectat {num_classes} clase în setul de test.")
    # -----------------------------------------------------------------------------

    # 2. Setare format PyTorch (Acum putem converti datele în tensori)
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
    model.eval() # IMPORTANT: Modul de evaluare

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

    # 5. Calcul Metrici
    acc = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    precision_macro = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, predictions, average='macro', zero_division=0)
    
    print(f"\n{'='*50}")
    print(f"\nREZULTATE FINALE PENTRU {task.upper()}:")
    print(f"Acuratețe (Accuracy): {acc:.4f}")
    print("-" * 30)
    print(f"F1-Score (Macro):     {f1_macro:.4f}")
    print(f"Precision (Macro):    {precision_macro:.4f}")
    print(f"Recall (Macro):       {recall_macro:.4f}")
    print("-" * 50)
    
    # 6. Metrici per Clasă
    # Numele claselor
    if task == 'sentiment':
        if num_classes == 2:
            target_names = ['Negativ', 'Pozitiv']
        elif num_classes == 3:
            target_names = ['Negativ', 'Pozitiv', 'Neutru']
        else:
            target_names = [f'Clasa_{i}' for i in range(num_classes)]
    else:
        target_names = [
            "forum",           # ID 0
            "promotions",      # ID 1
            "social_media",    # ID 2
            "spam",            # ID 3
            "updates",         # ID 4
            "verify_code"      # ID 5
        ]
        target_names = target_names[:num_classes]
    # Classification Report (conține precision, recall, f1 per clasă)
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=target_names, 
        digits=4,
        output_dict=True  # Returnează dict pentru a salva în JSON
    )
    
    print("\nRAPORT DETALIAT PER CLASĂ:")
    print(classification_report(
        true_labels, 
        predictions, 
        target_names=target_names, 
        digits=4,
        output_dict=True
    ))
    
    # 7. Matrice de Confuzie
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("-" * 50)
    print("Matrice de Confuzie (Rânduri=Real, Coloane=Predis):")
    print(conf_matrix)
    print("-" * 50)

    # 8. Pregătire date pentru salvare
    # Extragem metricile per clasă din report
    class_metrics = {}
    for class_name in target_names:
        if class_name in report:
            class_metrics[class_name] = {
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1-score': report[class_name]['f1-score'],
                'support': int(report[class_name]['support'])
            }
    
    # 9. Încărcăm metricile existente din train.py
    logger = MetricsLogger()
    existing_metrics = logger.load_metrics(task)
    run_id = None
    if existing_metrics and "run_id" in existing_metrics:
      run_id = existing_metrics["run_id"]

    if existing_metrics is None:
        print("⚠ ATENȚIE: Nu există metrici de antrenare. Se vor salva doar metricile de test.")
        existing_metrics = {}
    
    # 10. Adăugăm metricile de test la cele existente
    existing_metrics['test_results'] = {
        'accuracy': float(acc),
        'f1_score_macro': float(f1_macro),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'confusion_matrix': conf_matrix.tolist(),  # Convertim la listă pentru JSON
        'num_test_samples': len(true_labels)
    }
    
    existing_metrics['class_metrics'] = class_metrics
    
    # Actualizăm și numele claselor în config (dacă există)
    if 'config' not in existing_metrics:
        existing_metrics['config'] = {}
    existing_metrics['config']['class_names'] = target_names
    
    # 11. Salvăm metricile actualizate
    print(f"\n📊 Salvare metrici complete (train + test)...")
    logger.save_metrics(task, existing_metrics, run_id=run_id)
    
    print(f"\n{'='*50}")
    print("✅ TESTARE FINALIZATĂ CU SUCCES!")
    print(f"📈 Toate metricile au fost salvate în 'metrics/{task}_metrics.json'")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="sentiment sau category")
    args = parser.parse_args()
    
    if args.task not in ['sentiment', 'category']:
        print("Eroare: Task-ul trebuie să fie 'sentiment' sau 'category'.")
    else:
        run_testing(args.task)