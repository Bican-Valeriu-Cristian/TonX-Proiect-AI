import torch
import torch.nn as nn
from transformers import DistilBertModel

class TaskClassifier(nn.Module):
    def __init__(self, num_classes, model_name="distilbert-base-uncased", dropout=0.3):
        super(TaskClassifier, self).__init__()
        
        # Încărcăm modelul de bază DistilBERT (pre-antrenat)
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        # Strat de Dropout pentru a preveni overfitting-ul (rețeaua să nu "tocească")
        self.drop = nn.Dropout(p=dropout)
        
        # Stratul final de clasificare
        # 768 este dimensiunea output-ului standard DistilBERT
        # num_classes se va schimba automat (2 pt sentiment, X pt categorii)
        self.out = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Trecem datele prin DistilBERT
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extragem starea ascunsă a token-ului [CLS] (primul token)
        # care reprezintă rezumatul propoziției
        cls_output = output.last_hidden_state[:, 0, :] 
        
        # Aplicăm dropout
        output = self.drop(cls_output)
        
        # Clasificare finală
        return self.out(output)