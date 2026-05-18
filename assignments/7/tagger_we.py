import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from morpho_dataset import MorphoDataset


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
    def __init__(self, forms_count, we_dim, rnn_hidden_dim, tags_count):
        super().__init__()
        self.embedding = nn.Embedding(forms_count, we_dim, padding_idx=MorphoDataset.PAD)
        self.rnn = nn.LSTM(we_dim, rnn_hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(rnn_hidden_dim * 2, tags_count)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return logits.transpose(1, 2)


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    morpho = MorphoDataset("czech_cac")

    forms_to_index, tags_to_index, forms_count, tags_count = build_vocabularies(morpho)
    print(f"Forms count: {forms_count}, Tags count: {tags_count}")

    transform = make_transform(forms_to_index, tags_to_index)
    train_loader = DataLoader(morpho.train.with_transform(transform), batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=True)
    dev_loader = DataLoader(morpho.dev.with_transform(transform), batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=False)

    model = SimpleNN(forms_count, args.we_dim, args.rnn_hidden_dim, tags_count).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_dev_acc = 0.0

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for forms, tags in train_loader:
            forms, tags = forms.to(device), tags.to(device)
            
            # Word Dropout to heavily train the UNK token handling
            mask = torch.rand(forms.shape, device=device) < 0.10
            mask &= (forms != MorphoDataset.PAD)
            forms[mask] = MorphoDataset.UNK
            
            optimizer.zero_grad()
            outputs = model(forms)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_time = time.time() - start_time

        model.eval()
        dev_correct = 0
        dev_total = 0
        dev_loss = 0.0

        with torch.no_grad():
            for forms, tags in dev_loader:
                forms, tags = forms.to(device), tags.to(device)
                outputs = model(forms)
                loss = criterion(outputs, tags)
                dev_loss += loss.item()

                preds = outputs.argmax(dim=1)
                mask = tags != MorphoDataset.PAD
                dev_correct += (preds[mask] == tags[mask]).sum().item()
                dev_total += mask.sum().item()

        dev_acc = 100.0 * dev_correct / dev_total
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Dev Loss: {dev_loss/len(dev_loader):.4f} | "
              f"Dev Acc: {dev_acc:.2f}% | "
              f"Time: {train_time:.1f}s")

        if dev_acc > best_dev_acc:
            print(f"--> New best accuracy! {best_dev_acc:.2f}% -> {dev_acc:.2f}%. Saving weights...")
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), args.model_weights)

    print(f"Training completed. Best Dev Accuracy: {best_dev_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--we_dim", type=int, default=128)
    parser.add_argument("--rnn_hidden_dim", type=int, default=512)
    parser.add_argument("--model_weights", type=str, default="model_weights.tar")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
