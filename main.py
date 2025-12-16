import os 
from src.preprocessing_category import prepare_local_splits
from src.preprocessing_sentiment import make_clean_csv
from src.vectorization import run_tokenization_for_task 
from src.vectorization import TOKENIZED_OUTPUT_DIR 

def main():
  
    # 1. Ne asiguram ca directorul 'data/' exista
    data_dir = "data"
    # Verifică dacă directorul 'data/' există
    if not os.path.exists(data_dir):
        # Dacă nu există, îl creează
        os.makedirs(data_dir)
        print(f"Directorul '{data_dir}/' a fost creat.")

    # Ne asiguram ca exista si directorul pentru datele tokenizate
    # Verifică dacă directorul de output pentru tokenizare există
    if not os.path.exists(TOKENIZED_OUTPUT_DIR):
        # Dacă nu există, îl creează (exist_ok=True previne eroarea dacă există deja)
        os.makedirs(TOKENIZED_OUTPUT_DIR, exist_ok=True)
        print(f"Directorul pentru output tokenizat '{TOKENIZED_OUTPUT_DIR}/' a fost creat.")

    # --- Rulare preprocesare Categorie ---
    print("\n==============================================")
    print("INCEPERE PREPROCESARE: Categorie (Jason23322)")
    print("==============================================")
    try:
        # Etapa 1: Preprocesarea datelor Categorie (curățare, împărțire în seturi)
        prepare_local_splits()
        print("Preprocesare Categorie finalizata cu succes.")
        
        # === NOU: Rulare Tokenizare Categorie ===
        print("\n---  INCEPERE TOKENIZARE: Categorie ---")
        # Etapa 2: Tokenizarea datelor Categorie (conversia text -> ID-uri numerice)
        run_tokenization_for_task(task="category")
        print(" Tokenizare Categorie finalizata.")
        
    except Exception as e:
        # Afișează orice eroare apărută în pipeline-ul Categorie
        print(f" Eroare la preprocesarea/tokenizarea Categoriei: {e}")

    # --- Rulare preprocesare Sentiment ---
    print("\n===============================================")
    print("INCEPERE PREPROCESARE: Sentiment (S140 + Kaggle)")
    print("===============================================")
    try:
        # Etapa 1: Preprocesarea datelor Sentiment (unificare, curățare)
        make_clean_csv()
        print("Preprocesare Sentiment finalizata cu succes.")
        
        # === Rulare Tokenizare Sentiment ===
        print("\n---  INCEPERE TOKENIZARE: Sentiment ---")
        # Etapa 2: Tokenizarea datelor Sentiment
        run_tokenization_for_task(task="sentiment")
        print(" Tokenizare Sentiment finalizata.")
        
    except Exception as e:
        # Afișează orice eroare apărută în pipeline-ul Sentiment
        print(f" Eroare la preprocesarea/tokenizarea Sentimentului: {e}")

    print("\n\n===  TOATE PREGĂTIRILE SUNT FINALIZATE!  ===")


if __name__ == "__main__":
    main()