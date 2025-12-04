from src.preprocessing import make_clean_csv
from src.vectorization import tokenize_data

def main():
   # print("--- 1. Curățăm datele și facem split în train / test ---")
   # make_clean_csv()
    
    print("\n--- 2. Tokenizarea textului (DistilBERT) ---")
    tokenize_data()
    

if __name__ == "__main__":
    print("Incep preprocesarea și tokenizarea datelor...")
    main()
    print("\n--- Tokenizarea finalizată! Datele sunt pregătite pentru antrenare. ---")