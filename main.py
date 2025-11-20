print("PASUL 1: ÎNCĂRCARE DATASET-URI")

import load_dataset
print(" Dataset-uri încărcate!")

print("PASUL 2: CURĂȚARE DATASET-URI")

import clean_ds
print("Dataset-uri curățate!")

print("PASUL 3: UNIRE & SALVARE")


import merge_dataset
print(" Dataset-uri unite și salvate!")

print(" PIPELINE FINALIZAT CU SUCCES!")


print("\n Verifică rezultatele în:")
print("    data/processed/merged_dataset.csv")
print("    data/processed/merged_dataset_stats.txt")
