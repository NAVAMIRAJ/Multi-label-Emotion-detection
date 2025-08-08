from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizer, RobertaModel

tokenizer_name = "roberta-base"
model_path = "bert_emotion_best.pt"
max_len = 128
batch_size = 1
threshold = 0.5
emotions = ["anger", "fear", "joy", "sadness", "surprise"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

class EmotionDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx],
                        truncation=True,
                        padding="max_length",
                        max_length=max_len,
                        return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['texts'] = self.texts[idx]
        return item



class EmotionClassifier(nn.Module):
    def __init__(self, num_labels=5):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(tokenizer_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return self.classifier(pooled)
    

model = EmotionClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

@torch.no_grad()
def predict(csv_path):

    df = pd.read_csv(csv_path)
    print('Read the csv file')
    texts = df['text'].tolist()
    loader = DataLoader(EmotionDataset(texts), batch_size=batch_size)
    print("Loaded the Dataloader for inference")

    preds = []
    for batch in tqdm(loader):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        probs = torch.sigmoid(output).cpu().numpy()
        preds.append(probs)
    preds = np.vstack(preds)
    return (preds > threshold).astype(int)
    
if __name__ == "__main__":


    predictions = predict(sys.argv[1])
    df = pd.read_csv(sys.argv[1])
    actual_text = df['text'].tolist()
    
    for text, pred in zip(actual_text, predictions):
        print(text, *pred, sep=',')

    res_df = pd.DataFrame(predictions, columns=emotions)
    res_df.insert(0, 'text', actual_text)
    res_df.to_csv('test_result.csv', index=False)
