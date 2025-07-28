from transformers import LayoutLMTokenizerFast, LayoutLMModel
from transformers import LayoutLMTokenizer, LayoutLMModel
import torch
import pandas as pd

import ast
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda")
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
model = model.to(device)


class BBoxDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        words = str(row.text).split()
        box = ast.literal_eval(row.bbox)
        tokens, token_boxes = [], []

        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            if len(tokens) + len(word_tokens) >= self.max_len - 2:  # Reserve space for CLS and SEP
                break
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))

        # Add special tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[0, 0, 0, 0]]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding_length = self.max_len - len(input_ids)

        # Padding
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        token_boxes += [[0, 0, 0, 0]] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "bbox": torch.tensor(token_boxes, dtype=torch.long),
            "id": row["id"],
            "level": row["level"]
        }


dataset = BBoxDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
embeddings = defaultdict(list)
labels = defaultdict(list)

model.eval()
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        bbox = batch["bbox"].to(device)
        bbox = torch.clamp(bbox, 0, 1000)

        outputs = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # shape: [batch_size, hidden_size]

        for i, cls_emb in enumerate(cls_embeddings):
            id_ = batch["id"][i]
            level = batch["level"][i]
            embeddings[id_].append(cls_emb.cpu().numpy())
            labels[id_].append(level)
import numpy as np

# Example sequence of length 30 tokens
def paddend(sequence):
    # Pad so total length is 300
    array = np.array(sequence)
    if len(sequence)>1000:
        return sequence[:1000]
    padding_length = 1000 - array.shape[0]
    
    # Pad rows at the bottom (right-padding), zero-fill
    padded_array = np.pad(array, pad_width=((0, padding_length), (0, 0)), mode='constant', constant_values=0)
    return padded_array

def pad_label(label, max_len=1000):
    # Ensure label is a list, not a numpy array
    label = list(label)
    label = le.transform(label)
    shape = len(label)
    if shape > max_len:
        label = label[:max_len]
    else:
        padding = np.array([4]*(max_len -shape))
        label = np.concatenate([label,padding])
    return np.array(label)
for i in range(len(y)):
    y[i] = pad_label(y[i])


def extract_text_features(text):
    words = text.split()
    num_chars = len(text)
    num_words = len(words)
    avg_word_len = sum(len(w) for w in words) / max(1, num_words)

    capital_ratio = sum(1 for c in text if c.isupper()) / max(1, num_chars)
    is_all_caps = int(text.isupper())

    has_colon_or_dot = int(':' in text or '.' in text)
    ends_with_punct = int(text.strip()[-1] in ".!?") if len(text.strip()) > 0 else 0
    starts_with_number = int(text.strip()[0].isdigit()) if len(text.strip()) > 0 else 0

    stopword_count = sum(1 for w in words if w.lower() in stop_words)
    stopword_ratio = stopword_count / max(1, num_words)

    return pd.Series({
        'num_chars': num_chars,
        'num_words': num_words,
        'avg_word_len': avg_word_len,
        'capital_ratio': capital_ratio,
        'is_all_caps': is_all_caps,
        'has_colon_or_dot': has_colon_or_dot,
        'ends_with_punct': ends_with_punct,
        'starts_with_number': starts_with_number,
        'stopword_ratio': stopword_ratio
    })

df_features = data['text'].apply(extract_text_features)
data = pd.concat([data, df_features], axis=1)
wordfeatures = data[[ 'num_chars','num_words','avg_word_len','capital_ratio','is_all_caps','has_colon_or_dot','ends_with_punct','starts_with_number','stopword_ratio',"id"]]
feature_cols = [col for col in wordfeatures.columns if col != 'id']

# Step 3: Group by 'id' and collect features into sequences (list of rows per group)
grouped = wordfeatures.groupby('id')[feature_cols].apply(lambda df: df.to_numpy()).reset_index(name='features')

# Step 4: Pad each group to length 1000 with -1
MAX_LEN = 1000
PAD_VALUE = -1

def pad_sequence(seq, max_len=MAX_LEN, pad_value=PAD_VALUE):
    length, dim = seq.shape
    if length >= max_len:
        return seq[:max_len]
    else:
        pad = np.full((max_len - length, dim), pad_value)
        return np.vstack([seq, pad])

# Step 5: Apply padding
grouped['padded_features'] = grouped['features'].apply(pad_sequence)

# Step 6: Convert to final NumPy array (num_docs, 1000, num_features)
final_array = np.stack(grouped['padded_features'].to_numpy())



class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        output_features = X[:,:,-9:]
        X = X[:,:,:-9]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.output = torch.tensor(output_features,dtype = torch.float32)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx],self.output[idx]

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)

train_ds = EmbeddingDataset(X_train, y_train)
test_ds = EmbeddingDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)
