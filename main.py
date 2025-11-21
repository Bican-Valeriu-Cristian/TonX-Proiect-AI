from src.preprocessing import make_clean_csv
import pandas as pd

def count_labels(file_path):
    """
    Citește un fișier CSV și numără aparițiile etichetelor 0, 1 și 2.
    """
    try:
        df = pd.read_csv(file_path)
        if 'label' not in df.columns:
            return {"Eroare": "Fișierul nu conține coloana 'label'."}

        label_counts = df['label'].value_counts().to_dict()
        final_counts = {
            0: label_counts.get(0, 0),  # Negativ
            1: label_counts.get(1, 0),  # Pozitiv
            2: label_counts.get(2, 0)   # Neutru
        }
        return final_counts
    except FileNotFoundError:
        return {"Eroare": f"Fișierul nu a fost găsit la calea specificată: {file_path}"}
    except Exception as e:
        return {"Eroare": f"A apărut o eroare la citirea fișierului: {e}"}

def display_counts(dataset_name, results):
    """
    Funcție ajutătoare pentru afișarea rezultatelor.
    """
    if "Eroare" in results:
        print(f"Eroare la {dataset_name}: {results['Eroare']}")
    else:
        print(f"\nDistribuția Etichetelor în {dataset_name}:")
        print(f"  Sentiment Negativ (0): {results[0]}")
        print(f"  Sentiment Pozitiv (1): {results[1]}")
        print(f"  Sentiment Neutru (2): {results[2]}")
        total = results[0] + results[1] + results[2]
        print(f"  Total Rânduri: {total}")
        

def main():
    # print("Curățăm datele și facem split în train / test...")
    # make_clean_csv()

    # 1. Numărăm setul de antrenament
    results_train = count_labels("data/train.csv")
    display_counts("Setul de Antrenament (train.csv)", results_train)
    
    # 2. Numărăm setul de test
    results_test = count_labels("data/test.csv")
    display_counts("Setul de Test (test.csv)", results_test)
    
    # 3. Numărăm setul de validare
    results_val = count_labels("data/validation.csv")
    display_counts("Setul de Validare (validation.csv)", results_val)


if __name__ == "__main__":
    main()