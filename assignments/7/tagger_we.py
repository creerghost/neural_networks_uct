#!/usr/bin/env python3
# file: tagger_we.py

import argparse
import datetime
import re
import os

# uncomment to run on CPU only
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

import time

from morpho_dataset import MorphoDataset

print("Num GPUs Available: ", torch.cuda.device_count())


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--rnn_cell", default="LSTM", choices=["LSTM", "GRU"], help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=512, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
parser.add_argument("--hidden_layer_dim", default=None, type=int, help="Final linear hidden layer dimension. 0 or None for no final hidden layer.")
parser.add_argument("--packed_sequences", action="store_true", help="Whether to use packed sequences for RNN instead of sequences padded to the longest sequence present in batch.")

def train_model(model, device, train_loader, dev_loader, metric, criterion, optimizer, scheduler, writer, init_epoch,
                epochs):

    # Transfer the constructed metric object to the target device so
    # that state variables are computed and stored on that device.
    metric = metric.to(device)

    # Training loop
    for training_epoch in range(epochs):
        epoch = init_epoch + training_epoch
        print(f'Epoch {epoch + 1}/{init_epoch + epochs}:')
        # Training
        start_time = time.time()
        # Set the training mode flag
        model.train()

        train_loss = 0
        # initialize metric
        metric.reset()

        num_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            
            # Word Dropout (10% UNK masking)
            if model.training:
                mask = torch.rand(images.shape, device=device) < 0.10
                mask &= (images != MorphoDataset.PAD)
                images = images.clone()
                images[mask] = MorphoDataset.UNK
                
            outputs = model(images)
            loss = criterion(outputs, labels)

            train_loss += loss.item()
            metric.update(outputs.detach(), labels)

            loss.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()

            batch_idx += 1
            # if batch_idx % 100 == 0:
            #     print(f"train loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

        train_loss /= num_batches
        train_time = time.time() - start_time
        train_acc = metric.compute().cpu()
        # print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - train_time: {train_time:.4f} ms')
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        # Validation
        start_time = time.time()
        # Set the evaluation mode flag.
        model.eval()
        with torch.no_grad():
            val_loss = 0
            metric.reset()
            num_batches = len(dev_loader)
            for batch_idx, (images, labels) in enumerate(dev_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                metric.update(outputs.detach(), labels)

                batch_idx += 1
                # if batch_idx % 100 == 0:
                #     print(f"val loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

        val_loss /= num_batches
        val_time = time.time() - start_time
        val_acc = metric.compute().cpu()
        # print(f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_time: {val_time:.4f} s')
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)

        print(
            f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - train_time: {train_time:.4f} s - val_loss: {val_loss:.2f} - val_acc: {val_acc:.4f} - val_time: {val_time:.4f} s')

    writer.flush()
    return (train_acc, val_acc)


def eval_model(model, device, test_loader, metric, criterion, writer, epoch):

    # Transfer the constructed metric object to the target device so
    # that state variables are computed and stored on that device.
    metric = metric.to(device)

    # Test
    start_time = time.time()
    # Set the evaluation mode flag
    model.eval()
    with torch.no_grad():
        test_loss = 0
        metric.reset()

        num_batches = len(test_loader)
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            metric.update(outputs.detach(), labels)
            batch_idx += 1
            if batch_idx % 100 == 0:
                print(f"test loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

    test_loss /= num_batches
    test_time = time.time() - start_time
    test_acc = metric.compute().cpu()
    print(f'test_loss: {test_loss:.4f} - test_acc: {test_acc:.4f} - test_time: {test_time:.4f} s')
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", test_acc, epoch)

    writer.flush()
    return test_acc

def predict_model(model, device, data_loader):

    # Prediction
    start_time = time.time()
    # Set the evaluation mode flag
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for (images, _) in data_loader:
            images = images.to(device)
            outputs = model(images)
            # returned in format [batch, num_tags, seq]

            # Move the result to cpu, the input batch could stay on device.
            predicted = torch.argmax(outputs.detach(), 1).cpu()

            # Split the batch back into individual sentences, while different batches
            # can have different longest sequence lengths
            for forms, preds in zip(images, predicted):
                real_len = (forms != MorphoDataset.PAD).sum().item()
                all_outputs.append(preds[:real_len])  # strip MorphoDataset.PAD

    predict_time = time.time() - start_time
    print(f'predict_time: {predict_time:.4f} s')
    return all_outputs

class SimpleNN(nn.Module):
    def __init__(self, forms_count, we_dim, rnn_cell, rnn_cell_dim, hidden_layer_dim, tags_count, use_packed_sequences):
        super().__init__()
        self.use_packed_sequences = use_packed_sequences

        self.word_embedding = nn.Embedding(forms_count, we_dim, padding_idx=MorphoDataset.PAD)

        rnn_class = nn.LSTM if rnn_cell == "LSTM" else (nn.GRU if rnn_cell == "GRU" else nn.RNN)
        self.word_rnn = rnn_class(we_dim, rnn_cell_dim, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)

        self.rnn_cell_dim = rnn_cell_dim

        if hidden_layer_dim and hidden_layer_dim > 0:
            self.hidden = nn.Sequential(
                nn.Linear(rnn_cell_dim, hidden_layer_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        else:
            self.hidden = None

        final_dim = hidden_layer_dim if hidden_layer_dim and hidden_layer_dim > 0 else rnn_cell_dim
        self.fc_output = nn.Linear(final_dim, tags_count)

    def forward(self, word_ids_padded):
        x = self.word_embedding(word_ids_padded)

        if self.use_packed_sequences:
            lengths = (word_ids_padded != MorphoDataset.PAD).sum(dim=1).cpu()
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        x, _ = self.word_rnn(x)

        if self.use_packed_sequences:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        fwd = x[:, :, :self.rnn_cell_dim]
        bwd = x[:, :, self.rnn_cell_dim:]
        x = fwd + bwd

        if self.hidden is not None:
            x = self.hidden(x)

        x = self.fc_output(x)
        x = x.transpose(1, 2)

        return x

def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    # import random
    # random.seed(args.seed)
    # import numpy as np
    # np.random.seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))


    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # TODO: Construct python dictionary to translate form -> idx.
    #  Preserve indices 0 for padding and 1 for unknown in input form sequences.
    #  Use `word_mapping_vocabulary` of `morpho.train.forms`

    forms_to_index = {s: i + 2 for i, s in enumerate(morpho.train.forms.word_mapping_vocabulary)}
    forms_count = len(forms_to_index) + 2

    # TODO: Construct python dictionary to translate tag -> idx.
    #  Preserve index 0 for padding and 1 for unknown (either missing or in other than
    #  training datasets) in tag sequences. Use `word_mapping_vocabulary` of `morpho.train.tags`.

    tags_to_index = {t: i + 2 for i, t in enumerate(morpho.train.tags.word_mapping_vocabulary)}
    tags_count = len(tags_to_index) + 2

    def prepare_tagging_data_transform(sentence):
        # TODO: For given sentence (a dictionary having "forms", "lemmas", and
        #  "tags" lists), construct a training data consisting of the following pair (tuple):
        #  - a PyTorch tensor of integer ids of input forms as input,
        #  - a PyTorch tensor of integer tag ids as targets.
        #  For transformation, use constructed `forms_to_index` and `tags_to_index`.
        #  If the form is not found in the dictionary return 1 (`MorphoDataset.UNK`).
        #  If the tag is not found in the dictionary return 1 (`MorphoDataset.UNK`).
        #  Remember: If you make some evaluation with unknown tag indexed with a value
        #   outside the output dimension length, CUDA index check will probably fail
        #   and will report an error. In that case set the environment variable
        #   CUDA_LAUNCH_BLOCKING to "1" to throw the error immediately.

        form_ids = torch.tensor([forms_to_index.get(w, MorphoDataset.UNK) for w in sentence["forms"]], dtype=torch.long)
        
        tag_ids = torch.tensor([tags_to_index.get(t, MorphoDataset.UNK) for t in sentence["tags"]], dtype=torch.long)

        return form_ids, tag_ids

    train_transformed_dataset = morpho.train.with_transform(prepare_tagging_data_transform)
    dev_transformed_dataset = morpho.dev.with_transform(prepare_tagging_data_transform)
    test_transformed_dataset = morpho.test.with_transform(prepare_tagging_data_transform)

    # Construct a single batch from given `data`, where `data` is a list of transformed sentences
    # generated by `prepare_tagging_data_transform`, i.e. one pair of form_ids_padded tensor
    # and tag_ids_padded tensor for each sentence.
    def prepare_batch(data):

        form_ids, tag_ids = zip(*data)

        # TODO: Combine `form_ids` tuple into a single tensor, padding shorter
        #  sequences to length of the longest sequence in the batch with zeros (MorphoDataset.PAD)
        #  using `torch.nn.utils.rnn.pad_sequence` with `batch_first=True` argument
        #  (Remember: zero index in `forms_to_index` means padding).
        form_ids_padded = nn.utils.rnn.pad_sequence(form_ids, batch_first=True, padding_value=MorphoDataset.PAD)

        # TODO: Process `tag_ids` analogously to `form_ids`.
        tag_ids_padded = nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=MorphoDataset.PAD)

        return form_ids_padded, tag_ids_padded

    # Create the data
    train_loader = torch.utils.data.DataLoader(train_transformed_dataset, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_transformed_dataset, batch_size=args.batch_size, collate_fn=prepare_batch)
    test_loader = torch.utils.data.DataLoader(test_transformed_dataset, batch_size=args.batch_size, collate_fn=prepare_batch)

    for X, y in train_loader:
        print(f"Shape and type of sequences([B, seq_len]): {X.shape}, {X.dtype}")
        print(f"Shape and type of labels: {y.shape}, {y.dtype}")
        break

    # Get cpu, gpu or mps device for training
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Create the model.
    model = SimpleNN(forms_count,
                     args.we_dim,
                     args.rnn_cell,
                     args.rnn_cell_dim,
                     args.hidden_layer_dim,
                     tags_count,
                     args.packed_sequences)
    # Note: Thus there is a also an output for MorphoDataset.UNK tag, but we will ignore it.
    # And its value will probably be forced to be as small as possible.
    print(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('==================')
    print(f'Total parameters:{pytorch_total_params}')
    print(f'Trainable parameters:{pytorch_trainable_params}')
    print(f'Non-trainable parameters:{pytorch_total_params - pytorch_trainable_params}')
    print('==================')

    model.to(device)

    # TODO: Create the Adam optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TODO: Use the usual `torch.nn.CrossEntropyLoss` loss function. Additionally,
    #  pass `ignore_index=MorphoDataset.PAD` to the constructor so that the padded
    #  tags are ignored during the loss computation. Note that the loss
    #  expects the input to be of shape `[batch, num_tags, seq]`.
    criterion = nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD)
    
    # TODO: Create a `torchmetrics.Accuracy` metric, passing "multiclass" as
    #  the first argument, `num_classes` set to the number of unique tags, and
    #   again `ignore_index=MorphoDataset.PAD` to ignore the padding tags.

    metric = torchmetrics.Accuracy(task="multiclass", num_classes=tags_count, ignore_index=MorphoDataset.PAD)

    # TensorBoard writer initialization
    writer = SummaryWriter(logdir)

    init_epoch=0

    # Training loop
    train_acc, val_acc = train_model(model,
                device,
                train_loader,
                dev_loader,
                metric,
                criterion,
                optimizer,
                scheduler,
                writer,
                init_epoch,
                args.epochs)

    print(f"Model accuracies on train/dev: {train_acc:.4f}/{val_acc:.4f}", flush=True)


    predictions_sentences = predict_model(model, device,test_loader)

    # TODO: Construct inverted lookup table for idx -> tag.
    #  This should make inverse transformation to that handled with `tags_to_index`.
    #  For RNN outputs, for index 0 use "<PAD>", for 1 use "<UNK>". 
    #  Finally convert the list to numpy array.
    id_to_tag = np.asarray(["<PAD>", "<UNK>"] + morpho.train.tags.word_mapping_vocabulary)

    with open(os.path.join(logdir, "tagged.txt"), "w", encoding="utf-8") as predictions_file:
        print("word\tpredicted_tag\toriginal_label", file=predictions_file)

        for tested_sentences, pred_sentence_tag_ids in zip(
                morpho.test, predictions_sentences):
            forms = tested_sentences["forms"]
            tags = tested_sentences["tags"]
            # Convert from tensor to numpy array of integers.
            pred_sentence_tag_ids = np.asarray(pred_sentence_tag_ids, dtype=np.int64)
            predicted_tags = id_to_tag[pred_sentence_tag_ids]
            for form, predicted_tag, orig_label in zip(forms, predicted_tags, tags):
                print(form + "\t" + predicted_tag + "\t" + orig_label, file=predictions_file)
            print(file=predictions_file)

    test_acc = eval_model(model, device, test_loader, metric, criterion, writer, args.epochs-1)
    print(f"Model accuracy on test: {test_acc:.4f}", flush=True)

    writer.flush()
    writer.close()

    # TODO: Save model weights if you need the trained model later.
    # Remember also all the parameters used to construct the network with the exactly same architecture.    
    torch.save(model.state_dict(), "model_weights.tar")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
