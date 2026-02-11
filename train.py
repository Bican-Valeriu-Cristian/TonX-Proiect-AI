import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#  Importăm AdamW din PyTorch, nu din transformers ---
from torch.optim import AdamW 
from transformers import get_linear_schedule_with_warmup

from datasets import load_from_disk
from sklearn.utils.class_weight import compute_class_weight

# Importăm modelul definit în src/model.py
from src.model import TaskClassifier 

###
from tqdm import tqdm

# --- CONFIGURĂRI ---
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### 
def get_optimal_num_workers():
    try:
        # Numărul REAL de CPU-uri disponibile 
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # fallback dacă sistemul nu suportă sched_getaffinity
        available_cpus = os.cpu_count()

    # Regula optimă pentru CPU-only:
    # - dacă ai 1 CPU → 0 workers
    # - altfel → nr workers = nr de CPU - 1
    if available_cpus <= 1:
        return 0
    return max(1, available_cpus - 1)


def get_sklearn_class_weights(labels):
    """
    Calculează ponderile folosind Scikit-Learn.
    Primește direct un array numpy de etichete.
    """
    classes = np.unique(labels)
    
    # Calcul automat ("balanced" aplică formula inversă frecvenței)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    
    print(f"   Clase detectate: {classes}")
    print(f"   Ponderi calculate (sklearn): {weights}")
    
    # Returnăm un Tensor PyTorch (float) trimis pe GPU/CPU
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    ### 
    progress_bar = tqdm(
        data_loader,
        desc="Training",
        total=len(data_loader),
        leave=False
    )

    ### 
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 1. Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. Calcul predicții și eroare
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        # 3. Backward pass (Învățare)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Evită explozia gradienților
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        ###
        # live updates in progress bar
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{(correct_predictions.double() / n_examples):.4f}"
        )

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval() # Dezactivează Dropout
    losses = []
    correct_predictions = 0

    ### 
    progress_bar = tqdm(
        data_loader,
        desc="Validation",
        total=len(data_loader),
        leave=False
    )

    with torch.no_grad():  # Nu calculăm gradienți (economie memorie/timp)
        ### 
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            ### 
            # update progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(correct_predictions.double() / n_examples):.4f}"
            )

    return correct_predictions.double() / n_examples, np.mean(losses)

def run_training(task):
    print(f"\n=======================================================")
    print(f" PORNIRE ANTRENARE | TASK: {task.upper()}")
    print(f"=======================================================")
    
    # 1. Încărcare date
    data_path = os.path.join("data", "tokenized_datasets", task)
    if not os.path.exists(data_path):
        print(f"EROARE: Nu găsesc datele în '{data_path}'. Rulează 'python main.py' mai întâi!")
        return

    dataset_dict = load_from_disk(data_path)
    train_data = dataset_dict['train']
    val_data = dataset_dict['validation']

    #  Calculăm clasele și ponderile ÎNAINTE de conversia la PyTorch ---
    # Convertim coloana 'label' într-un numpy array simplu
    print(" INFO: Analiză distribuție clase...")
    all_labels = np.array(train_data['label']) 
    num_classes = len(set(all_labels))
    print(f" INFO: S-au detectat {num_classes} clase unice.")

    # Calcul Ponderi
    class_weights = get_sklearn_class_weights(all_labels)
    # -------------------------------------------------------------------------------

    # 2. Setare format PyTorch (abia acum transformăm datele în Tensors)
    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,

        ### 
        num_workers=get_optimal_num_workers(),
        persistent_workers=False,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,

        ###
        num_workers=get_optimal_num_workers(),
        persistent_workers=False,
        pin_memory=False
    )

    # 3. Configurare Model
    model = TaskClassifier(num_classes=num_classes)
    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 4. Bucla de antrenare
    best_accuracy = 0
    save_path = os.path.join("models", f"{task}_best_model.bin")
    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print('-' * 20)

        train_acc, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, DEVICE, scheduler, len(train_data)
        )
        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        val_acc, val_loss = eval_model(
            model, val_loader, loss_fn, DEVICE, len(val_data)
        )
        print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), save_path)
            best_accuracy = val_acc
            print(f" >> Model Nou Salvat! ({save_path})")

    print(f"\n Gata! Modelul final pentru {task} este salvat.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="sentiment sau category")
    args = parser.parse_args()
    
    if args.task not in ['sentiment', 'category']:
        print("Eroare: Task-ul trebuie să fie 'sentiment' sau 'category'.")
    else:
        run_training(args.task)