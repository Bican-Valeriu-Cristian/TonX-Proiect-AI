import torch
import os
import json
import numpy as np
from transformers import DistilBertTokenizer
from src.model import TaskClassifier
from src.preprocessing import simple_clean

# Configurare dispozitiv
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TonXPredictor:
    def __init__(self, task, is_raw=False):
        """
        Inițializează predictorul pentru un anumit task ('sentiment' sau 'category').
        Încarcă modelul antrenat și configurația claselor.
        """
        self.task = task
        self.is_raw = is_raw
        self.model = None
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.class_names = []
        self.ready = False

        # Căi către fișiere
        self.model_path = os.path.join("models", f"{task}_best_model.bin")
        self.metrics_path = os.path.join("metrics", f"{task}_metrics.json")

        self._load_resources()

    def _load_resources(self):
        # 1. Determinăm numărul de clase
        # Default de siguranță
        num_classes = 2 
        
        # LOGICĂ MANUALĂ (Hardcodată pentru siguranță):
        if self.task == "sentiment":
            num_classes = 3
        elif self.task == "category":
            num_classes = 6  # <--- MODIFICAREA PENTRU CATEGORIE (bazat pe eroare)
        
        # Încercăm să citim din metrics.json pentru nume de clase
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    data = json.load(f)
                    config = data.get('config', {})
                    loaded_names = config.get('class_names', [])
                    
                    if loaded_names:
                        self.class_names = loaded_names
                        # Dacă JSON-ul are nume, actualizăm și numărul (pentru consistență)
                        if len(self.class_names) > 0:
                             num_classes = len(self.class_names)

            except Exception as e:
                print(f"⚠️ Avertisment: Nu s-a putut încărca config-ul pentru {self.task}: {e}")
        
        # Dacă nu avem nume de clase (JSON lipsă), le generăm generic
        if not self.class_names:
            if self.task == 'sentiment':
                self.class_names = ['Negativ', 'Pozitiv', 'Neutru']
            elif self.task == 'category':
                # Nume generice pentru cele 6 clase (le poți schimba manual aici dacă știi ordinea)
                self.class_names = [f"Categorie_{i}" for i in range(num_classes)]
            else:
                self.class_names = [f"Clasa_{i}" for i in range(num_classes)]

        # 2. Inițializăm și încărcăm modelul
        print(f"🔄 Încărcare model {self.task} ({num_classes} clase)...")
        try:
            # Inițializăm arhitectura cu numărul corect de clase
            self.model = TaskClassifier(num_classes=num_classes).to(DEVICE)
            if not self.is_raw: 
                if os.path.exists(self.model_path):
                    # Încărcăm greutățile cu strict=False pentru a fi mai permisivi, dar acum dimensiunile sunt corecte
                    self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
                    print(f"✅ Model {self.task} încărcat cu succes!")
                    self.ready = True
                else:
                    print(f"❌ Eroare: Nu s-a găsit fișierul modelului la {self.model_path}")
                    self.ready = False
            else:
                    print(f"ℹ️ Modelul {self.task} este în stare RAW (neantrenat).")
                    self.ready = True
            self.model.eval()
        
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"❌ EROARE DIMENSIUNI: Codul așteaptă {num_classes} clase, dar modelul salvat are alt număr.")
                print(f"   -> Verifică linia 'num_classes = ...' în predict.py.")
            print(f"Eroare detaliată: {e}")
        except Exception as e:
            print(f"❌ Eroare critică la încărcarea modelului: {e}")

    def predict(self, text):
        """
        Primește un text, îl preprocesează și returnează:
        (etichetă_text, probabilitate, index_clasă)
        """
        if not self.ready:
            return "Model Neloadat", 0.0, -1

        # 1. Preprocesare
        clean_text = simple_clean(text)

        # 2. Tokenizare
        encoding = self.tokenizer(
            clean_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        # 3. Inferență
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            # Aplicăm Softmax pentru a obține probabilități (0-1)
            probs = torch.softmax(outputs, dim=1)
            
            # Luăm clasa cu probabilitatea cea mai mare
            max_prob, preds = torch.max(probs, dim=1)
            
            class_idx = preds.item()
            confidence = max_prob.item()

        # 4. Mapare la etichetă
        if class_idx < len(self.class_names):
            label = self.class_names[class_idx]
        else:
            label = f"Clasa {class_idx}"

        return label, confidence, class_idx