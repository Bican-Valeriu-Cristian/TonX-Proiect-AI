import pandas as pd
import numpy as np

def compute_class_weights(train_path="data/train.csv"):
    df_train = pd.read_csv("../data/train.csv")
    labels = df_train["label"].values

    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    num_classes = len(classes)

    # formula clasica: w_c = N / (num_classes * count_c)
    class_weights = {
        int(c): float(total) / (num_classes * float(cnt))
        for c, cnt in zip(classes, counts)
    }

    print("Distributie clase:", dict(zip(classes, counts)))
    print("Class weights:", class_weights)
    return class_weights

if __name__ == "__main__":
    compute_class_weights()

# --- test---