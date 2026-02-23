import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# IMPORTANT: folosește același model/forward ca la training (în proiectul tău e în src/model.py)
from src.model import TaskClassifier
MODEL_NAME = "distilbert-base-uncased"  # sau exact ce ai folosit tu la training
from src.model import TaskClassifier
MODEL_NAME = "distilbert-base-uncased"  # sau exact ce ai folosit tu la training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(task: str, num_classes: int):
    weights_path = os.path.join("models", f"{task}_best_model.bin")

    print(f"Încărcare model din {weights_path}...")
    model = TaskClassifier(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    # tokenizerul trebuie să fie identic cu cel folosit în training
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("✅ Model încărcat!\n")
    return model, tokenizer


@torch.no_grad()
def predict(text, model, tokenizer, labels):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    probs = F.softmax(logits, dim=-1)

    pred_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_class].item()

    return labels[pred_class], confidence, probs[0]


def main():
    print("=" * 60)
    print("🤖 TonX Model Tester (din .bin)")
    print("=" * 60)
    print("\nAlege task-ul:")
    print("1. Sentiment")
    print("2. Category")
    print("3. Ambele")

    choice = input("\nOpțiune (1/2/3): ").strip()

    models = {}

    # Ajustează label-urile dacă la tine ordinea e alta
    if choice in ["1", "3"]:
        sentiment_labels = ["Negativ", "Pozitiv", "Neutru"]
        models["sentiment"] = {
            "model": load_model("sentiment", num_classes=len(sentiment_labels)),
            "labels": sentiment_labels,
        }

    if choice in ["2", "3"]:
        # IMPORTANT: aici trebuie să pui exact numărul și numele claselor tale
        category_labels =["forum", "promotions", "social_media", "spam", "updates", "verify_code"]
        models["category"] = {
            "model": load_model("category", num_classes=len(category_labels)),
            "labels": category_labels,
        }

    print("\n" + "=" * 60)
    print("Introduceți textul de testat (sau 'exit' pentru ieșire)")
    print("=" * 60)

    while True:
        text = input("\n> ")
        if text.lower() == "exit":
            break
        if not text.strip():
            continue

        for task_name, task_data in models.items():
            model, tokenizer = task_data["model"]
            labels = task_data["labels"]

            prediction, confidence, all_probs = predict(text, model, tokenizer, labels)

            print(f"\n{'=' * 60}")
            print(f"📊 {task_name.upper()}")
            print(f"{'=' * 60}")
            print(f"✅ Predicție: {prediction}")
            print(f"📈 Încredere: {confidence:.2%}")
            print(f"\n📊 Toate probabilitățile:")

            for i, label in enumerate(labels):
                prob = all_probs[i].item()
                bar = "█" * int(prob * 40)
                print(f"  {label:12} | {bar} {prob:.2%}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 La revedere!")
        sys.exit(0)
