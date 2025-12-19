import torch
import os
from transformers import DistilBertTokenizerFast
from src.model import TaskClassifier

# Maparea label-urilor pentru task-ul Categorie (din dataset-ul Jason23322)
CATEGORY_MAP = {
    0: "Forum - Forum posts, discussions, and community notifications",
    1: "Promotions - Marketing emails, sales, offers, and advertisements",
    2: "Social Media - Notifications from social platforms",
    3: "Spam - Unwanted emails, scams, and phishing attempts",
    4: "Updates -System updates, security patches, maintenance notices",
    5: "Verify Code - Authentication codes and verification emails"
}

def predict_category(text, model_path="models/category_best_model.bin"):
    """
    Efectuează o predicție reală folosind modelul antrenat.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n--- ANALIZĂ CATEGORIE PENTRU: '{text}' ---")

    if not os.path.exists(model_path):
        print(f" (Modelul antrenat nu a fost găsit la '{model_path}')")
        return None, 0

    # Dacă modelul există, facem predicție reală
    try:
        # Încărcăm modelul (presupunem 6 clase conform dataset-ului)
        model = TaskClassifier(num_classes=6)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_id = torch.max(probs, dim=1)
            
        label = CATEGORY_MAP.get(pred_id.item(), "Unknown")
        print(f" Rezultat Predicție (Reală): {label}")
        print(f" Încredere Model: {confidence.item()*100:.2f}%")
        return label, confidence.item()

    except Exception as e:
        print(f" Eroare la rularea modelului real: {e}")
        return None, 0
