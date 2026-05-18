import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from morpho_dataset import MorphoDataset
import numpy as np

def build_vocabularies(morpho):
    forms_to_index = {s: i + 2 for i, s in enumerate(morpho.train.forms.word_mapping_vocabulary)}
    forms_count = len(forms_to_index) + 2
    tags_to_index = {t: i + 2 for i, t in enumerate(morpho.train.tags.word_mapping_vocabulary)}
    tags_count = len(tags_to_index) + 2
    return forms_to_index, tags_to_index, forms_count, tags_count

def make_transform(forms_to_index, tags_to_index):
    def prepare_tagging_data_transform(sentence):
        form_ids = [forms_to_index.get(w, MorphoDataset.UNK) for w in sentence["forms"]]
        tag_ids = [tags_to_index.get(t, MorphoDataset.UNK) for t in sentence["tags"]]
        return torch.as_tensor(form_ids), torch.as_tensor(tag_ids)
    return prepare_tagging_data_transform

def prepare_batch(data):
    form_ids, tag_ids = zip(*data)
    form_ids_padded = nn.utils.rnn.pad_sequence(list(form_ids), batch_first=True, padding_value=MorphoDataset.PAD)
    tag_ids_padded = nn.utils.rnn.pad_sequence(list(tag_ids), batch_first=True, padding_value=MorphoDataset.PAD)
    return form_ids_padded, tag_ids_padded

class SimpleNN(nn.Module):
    def __init__(self, forms_count, tags_count):
        super().__init__()
        self.embedding = nn.Embedding(forms_count, 128, padding_idx=MorphoDataset.PAD)
        self.rnn = nn.LSTM(128, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(512, tags_count)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return logits.transpose(1, 2)

morpho = MorphoDataset("czech_cac")
forms_to_index, tags_to_index, forms_count, tags_count = build_vocabularies(morpho)
transform = make_transform(forms_to_index, tags_to_index)

train_loader = DataLoader(morpho.train.with_transform(transform), batch_size=64, collate_fn=prepare_batch, shuffle=True)
dev_loader = DataLoader(morpho.dev.with_transform(transform), batch_size=64, collate_fn=prepare_batch)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleNN(forms_count, tags_count).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training on", device)
for epoch in range(5):
    model.train()
    for forms, tags in train_loader:
        forms, tags = forms.to(device), tags.to(device)
        optimizer.zero_grad()
        loss = criterion(model(forms), tags)
        loss.backward()
        optimizer.step()
        
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for forms, tags in dev_loader:
            forms, tags = forms.to(device), tags.to(device)
            outputs = model(forms)
            preds = outputs.argmax(dim=1)
            mask = tags != MorphoDataset.PAD
            correct += (preds[mask] == tags[mask]).sum().item()
            total += mask.sum().item()
    print(f"Epoch {epoch+1} Dev Acc: {100*correct/total:.2f}%")
