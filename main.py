import os 
from src.preprocessing_category import prepare_local_splits
from src.preprocessing_sentiment import make_clean_csv
from src.vectorization import run_tokenization_for_task, TOKENIZED_OUTPUT_DIR, demonstrate_tokenization
from src.inference import predict_category

def main():
  
    # Exemplu Tokenizare
    print("\n" + "="*50)
    print("DEMONSTRAȚIE PENTRU TOKENIZARE")
    print("="*50)
    
    ex_text = "This project uses AI to analyze sentiments and categories."
    demonstrate_tokenization(ex_text)

    #Exemplu Clasificare
    print("\n" + "="*50)
    print("DEMONSTRAȚIE CLASIFICARE (CATEGORIE)")
    print("="*50)
    
    test_emails = [
        "Special offer! Get 50% discount on all our products today!",
        "Your verification code for TonX is: 554210. Do not share this code.",
        "Your friend tagged you in a new post on SocialMedia.",
        "software update available v confirm cancel company com updates address same building"
    ]
    
    for email in test_emails:
        predict_category(email)

    # # 1. Ne asiguram ca directorul 'data/' exista
    # data_dir = "data"
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir)
    #     print(f"Directorul '{data_dir}/' a fost creat.")

    # if not os.path.exists(TOKENIZED_OUTPUT_DIR):
    #     os.makedirs(TOKENIZED_OUTPUT_DIR, exist_ok=True)
    #     print(f"Directorul pentru output tokenizat '{TOKENIZED_OUTPUT_DIR}/' a fost creat.")

    # # --- Rulare preprocesare Categorie ---
    # print("\n==============================================")
    # print("INCEPERE PREPROCESARE: Categorie (Jason23322)")
    # print("==============================================")
    # try:
    #     prepare_local_splits()
    #     print("Preprocesare Categorie finalizata cu succes.")
        
    #     print("\n---  INCEPERE TOKENIZARE: Categorie ---")
    #     run_tokenization_for_task(task="category")
    #     print(" Tokenizare Categorie finalizata.")
        
    # except Exception as e:
    #     print(f" Eroare la preprocesarea/tokenizarea Categoriei: {e}")

    # # --- Rulare preprocesare Sentiment ---
    # print("\n===============================================")
    # print("INCEPERE PREPROCESARE: Sentiment (S140 + Kaggle)")
    # print("===============================================")
    # try:
    #     make_clean_csv()
    #     print("Preprocesare Sentiment finalizata cu succes.")
        
    #     print("\n---  INCEPERE TOKENIZARE: Sentiment ---")
    #     run_tokenization_for_task(task="sentiment")
    #     print(" Tokenizare Sentiment finalizata.")
        
    # except Exception as e:
    #     print(f" Eroare la preprocesarea/tokenizarea Sentimentului: {e}")

    # print("\n\n===  TOATE PREGĂTIRILE SUNT FINALIZATE!  ===")


if __name__ == "__main__":
    main()
