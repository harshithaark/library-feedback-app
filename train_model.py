import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight  # Correct import
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess data
data = pd.read_csv('data/train.csv')

# Drop rows with missing values
data = data.dropna(subset=['Review', 'Rating'])

# Ensure 'Rating' column is numeric and clean any non-numeric entries
data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')

# Drop rows where 'Rating' is NaN after conversion
data = data.dropna(subset=['Rating'])

# Convert 'Rating' to integer type
data['Rating'] = data['Rating'].astype(int)

# Map ratings to sentiment labels
rating_map = {1: 0, 3: 1, 5: 2}  # Negative: 0, Neutral: 1, Positive: 2
data = data[data['Rating'].isin(rating_map.keys())]
data['Rating'] = data['Rating'].map(rating_map)

# Check if the dataset is still valid after filtering
if len(data) == 0:
    print("No valid data available after filtering ratings. Exiting...")
    exit()

# Split into training and testing datasets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['Review'].tolist(),
    data['Rating'].tolist(),
    test_size=0.2,
    random_state=42
)

# Tokenization using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encodings = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3  # Negative, Neutral, Positive
)
model.to(device)  # Move the model to the device here

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=data['Rating'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training Loop
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Training Loss: {train_loss / len(train_loader):.4f}")

    # Validation Loop
    model.eval()
    val_preds = []
    val_labels_list = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_labels_list, val_preds)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the model
model.save_pretrained('model/')
tokenizer.save_pretrained('model/')
print("Model saved!")
