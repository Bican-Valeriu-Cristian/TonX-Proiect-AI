import pandas as pd  # type: ignore[import]
import config  # type: ignore
import os
from clean_ds import ds_sentiment140_clean, df_neutral_clean

# Adaugă coloană pentru a ști de unde vine fiecare rând
ds_sentiment140_clean['source'] = 'sentiment140'
df_neutral_clean['source'] = 'twitter_sentiment'

print(f" Sentiment140: {len(ds_sentiment140_clean)} rânduri")
print(f" Twitter Sentiment: {len(df_neutral_clean)} rânduri")

# Combină dataset-urile
datasets_to_merge = [ds_sentiment140_clean]

if len(df_neutral_clean) > 0:
    datasets_to_merge.append(df_neutral_clean)
    print(f" Unire 2 dataset-uri...")
else:
    print(f" Folosim doar Sentiment140 (nu există date neutre)...")

# Concatenează
df_merged = pd.concat(datasets_to_merge, ignore_index=True)

print(f" Dataset înainte de shuffle: {len(df_merged)} rânduri")

# Amestecă rândurile (shuffle)
df_merged = df_merged.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)

print(f" Dataset după shuffle: {len(df_merged)} rânduri")

# Adaugă label-uri text pentru claritate
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
df_merged['sentiment'] = df_merged['target'].map(label_map)

# Adaugă encoding pentru training
df_merged['label'] = df_merged['target']

print(f" Coloane finale: {df_merged.columns.tolist()}")
print(f"\n Statistici generale:")
print(f"   Total rânduri: {len(df_merged):,}")
print(f"   Total coloane: {len(df_merged.columns)}")

print(f"\n Distribuție pe clase:")
for sentiment in ['negative', 'neutral', 'positive']:
    count = len(df_merged[df_merged['sentiment'] == sentiment])
    percentage = (count / len(df_merged)) * 100
    print(f"   {sentiment.capitalize()}: {count:,} ({percentage:.2f}%)")

print(f"\n Distribuție pe surse:")
for source, count in df_merged['source'].value_counts().items():
    percentage = (count / len(df_merged)) * 100
    print(f"   {source}: {count:,} ({percentage:.2f}%)")

# Analiză dezechilibru
counts = df_merged['sentiment'].value_counts()
max_count = counts.max()
min_count = counts.min()
imbalance_ratio = max_count / min_count

print(f"\n Analiză dezechilibru:")
print(f"   Ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print(f"    Dezechilibru SEMNIFICATIV! Recomandare: balansare")
elif imbalance_ratio > 1.5:
    print(f"    Dezechilibru moderat")
else:
    print(f"    Dataset relativ echilibrat")

# Statistici lungime text
print(f"\n Statistici lungime text:")
print(f"   Medie: {df_merged['text'].str.len().mean():.2f} caractere")
print(f"   Median: {df_merged['text'].str.len().median():.2f} caractere")
print(f"   Min: {df_merged['text'].str.len().min()} caractere")
print(f"   Max: {df_merged['text'].str.len().max()} caractere")

# Creează director output dacă nu există
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
print(f" Director output: {config.OUTPUT_DIR}")

# Salvează dataset combinat
output_file = os.path.join(config.OUTPUT_DIR, 'merged_dataset.csv')
df_merged.to_csv(output_file, index=False, encoding='utf-8')
print(f" Dataset salvat: {output_file}")

# Salvează statistici
stats_file = os.path.join(config.OUTPUT_DIR, 'merged_dataset_stats.txt')
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("STATISTICI DATASET MERGED\n")
    
    f.write(f"Total samples: {len(df_merged):,}\n")
    f.write(f"Total coloane: {len(df_merged.columns)}\n")
    f.write(f"Coloane: {', '.join(df_merged.columns.tolist())}\n\n")
    
    f.write("Distribuție pe clase:\n")
    for sentiment in ['negative', 'neutral', 'positive']:
        count = len(df_merged[df_merged['sentiment'] == sentiment])
        percentage = (count / len(df_merged)) * 100
        f.write(f"  {sentiment}: {count:,} ({percentage:.2f}%)\n")
    
    f.write("\nDistribuție pe surse:\n")
    for source, count in df_merged['source'].value_counts().items():
        percentage = (count / len(df_merged)) * 100
        f.write(f"  {source}: {count:,} ({percentage:.2f}%)\n")
    
    f.write(f"\nRatio dezechilibru: {imbalance_ratio:.2f}:1\n")
    
    f.write("\nStatistici lungime text:\n")
    f.write(df_merged['text'].str.len().describe().to_string())
    f.write("\n")

print(f" Statistici salvate: {stats_file}")
print(f"\n Preview dataset final (primele 5 rânduri):")
print(df_merged[['text', 'sentiment', 'label', 'source']].head())

print(f"\n Preview dataset final (ultimele 5 rânduri):")
print(df_merged[['text', 'sentiment', 'label', 'source']].tail())

print(f"\nFișiere generate:")
print(f"   • {output_file}")
print(f"   • {stats_file}")
# Exportă pentru a fi folosit în alte scripturi
if __name__ == "__main__":
    print("ℹ Dataset disponibil în variabila: df_merged")