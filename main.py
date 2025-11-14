from src.preprocessing import make_clean_csv

def main():
    print("Curățăm datele și facem split în train / test...")
    make_clean_csv()

if __name__ == "__main__":
    print("Incep preprocesarea datelor...")
    make_clean_csv()
    print("Gata!")