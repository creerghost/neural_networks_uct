#!/usr/bin/env python3
# file: uppercase.py
import argparse
import datetime
import os
import re
# uncomment to run on CPU only
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

print("Num GPUs Available: ", torch.cuda.device_count())

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=None, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=None, type=int, help="Window size to use.")

def train_model(model, device, train_loader, dev_loader, optimizer, scheduler, criterion, writer, init_epoch, epochs):

    # Training loop
    train_acc, val_acc = 0, 0
    for training_epoch in range(epochs):
        epoch = init_epoch + training_epoch
        print(f'Epoch {epoch + 1}/{init_epoch + epochs}:')
        # Training
        start_time = time.time()
        # Set the training mode flag
        model.train()
        train_loss, train_correct = 0, 0
        num_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            train_loss += loss.item()
            predicted = torch.argmax(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            batch_idx += 1
            if batch_idx % 100 == 0:
                 print(f"train loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= num_batches
        train_time = time.time() - start_time
        # print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - train_time: {train_time:.4f} ms')
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        # Validation
        start_time = time.time()
        # Set the evaluation mode flag
        model.eval()
        with torch.no_grad():
            val_loss, val_correct = 0, 0
            num_batches = len(dev_loader)
            for batch_idx, (images, labels) in enumerate(dev_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = torch.argmax(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

                batch_idx += 1
                if batch_idx % 100 == 0:
                     print(f"val loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

        val_acc = val_correct / len(dev_loader.dataset)
        val_loss /= num_batches
        val_time = time.time() - start_time
        # print(f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_time: {val_time:.4f} s')
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)

        print(
            f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - train_time: {train_time:.4f} s - val_loss: {val_loss:.2f} - val_acc: {val_acc:.4f} - val_time: {val_time:.4f} s')

    writer.flush()
    return (train_acc, val_acc)

def eval_model(model, device, test_loader, criterion, writer, epoch):

    # Evaluation
    start_time = time.time()
    # Set the evaluation mode flag
    model.eval()
    with torch.no_grad():
        test_loss, test_correct = 0, 0
        num_batches = len(test_loader)
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            predicted = torch.argmax(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            batch_idx += 1
            if batch_idx % 100 == 0:
                print(f"test loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

    test_acc = test_correct / len(test_loader.dataset)
    test_loss /= num_batches
    test_time = time.time() - start_time
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
    all_outputs = torch.tensor([], device="cpu")
    with torch.no_grad():
        for (images, _) in data_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            predicted = predicted.to("cpu")
            all_outputs = torch.cat((all_outputs, predicted), 0)

    predict_time = time.time() - start_time
    print(f'predict_time: {predict_time:.4f} s')
    return all_outputs


# TODO: Implement a suitable model, optionally including regularization, select
#   good hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is represented by a `torch.int16` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit PyTorch layer, but you can use `torch.nn.functional.one_hot(inputs, alphabet_size_used)`.
#   However this function requires only torch.int64 for inputs. You can use `to(dtype=torch.int64)`
#   or `.long()` to convert it. Subsequent layers require float values, so after the conversion
#   to _one_hot encoding_, remember to convert values back to floats. You can then flatten
#   the one-hot encoded windows and follow with a linear/dense layer.
# - Alternatively, you can use `torch.nn.Embedding` (which is an efficient
#   implementation of one-hot encoding followed by a Linear layer) and flatten afterwards.

class SimpleNN(nn.Module):
    def __init__(self, input_size, alphabet_size_used, ...):
        super().__init__()
        self.alphabet_size_used = alphabet_size_used

        self.flatten = nn.Flatten()
        input_size *= alphabet_size_used

        ...
        # self.hidden =
        # input_size =


        self.fc_output = nn.Linear(input_size, 2)
        # self.softmax = nn.Softmax(dim=1) # dim=0 is for batch size, we do not need softmax, because we use nn.CrossEntropyLoss

    def forward(self, x):
        x = torch.nn.functional.one_hot(x.long(), self.alphabet_size).float()
        x = self.flatten(x)

        ...
        # x = self.hidden(x)


        x = self.fc_output(x)
        # x = self.softmax(x)
        return x


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    # import random
    # random.seed(args.seed)
    # import numpy as np
    np.random.seed(args.seed)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
    
    assert(args.batch_size is not None)
    
    # Create logdir name
    logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    assert (args.window is not None)
    uppercase_data = UppercaseData(args.window, args.alphabet_size)
    train_loader = DataLoader(uppercase_data.train, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(uppercase_data.dev, batch_size=args.batch_size, shuffle=False)

    for X, y in train_loader:
        print(f"Shape and type of windows: {X.shape}, {X.dtype}")
        print(f"Shape and type of labels: {y.shape}, {y.dtype}")
        break

    # Get cpu, gpu or mps device for training
    if args.device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    print(f"Using {device} device")

    # Note: To prevent problems with saving the model (serialization), save the alphabet size
    # to a simple python variable (as in the example above) and read this variable later in 
    # `torch.nn.functional.one_hot` function instead of reading directly the UppercaseData object 
    # properties.
    alphabet_size_used = len(uppercase_data.train.alphabet)

    model = SimpleNN(2 * args.window + 1, alphabet_size_used)


    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('==================')
    print(f'Total parameters:{pytorch_total_params}')
    print(f'Trainable parameters:{pytorch_trainable_params}')
    print(f'Non-trainable parameters:{pytorch_total_params - pytorch_trainable_params}')
    print('==================')

    model.to(device)

    # TODO: Select the loss and the optimizer
    ...
    # criterion = ...
    # optimizer = ...
    # scheduler = ...

    # TensorBoard writer initialization
    writer = SummaryWriter(logdir)

    init_epoch=0
    assert (args.epochs is not None)

    # Training loop
    train_acc, val_acc = train_model(model,
                device,
                train_loader,
                dev_loader,
                optimizer,
                scheduler,
                criterion,
                writer,
                init_epoch, args.epochs)

    writer.flush()
    writer.close()

    print(f"Model accuracies on train/dev: {train_acc:.4f}/{val_acc:.4f}", flush=True)

    # TODO: If your model is not too tied to other python objects, you can save it 
    # (serialize it) using `torch.save` function.
    ...

    # If something goes really wrong, save at least the model weights.
    
    #
    # Generate correctly capitalized dev and test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to predictions_file (which is
    # `uppercase_test.txt` in the `logdir` directory).
    #
    os.makedirs(logdir, exist_ok=True)

    dev_prediction = predict_model(model,device,dev_loader)
    with open(os.path.join(logdir, "uppercase_dev.txt"), "w", encoding="utf-8") as predictions_file:
        for character in range(dev_prediction.shape[0]):
            if dev_prediction[character] == 1:
                print(uppercase_data.dev.text[character].upper(), file=predictions_file, end='')
            else:
                print(uppercase_data.dev.text[character].lower(), file=predictions_file, end='')


    test_loader = DataLoader(uppercase_data.test, batch_size=args.batch_size, shuffle=False)
    test_prediction = predict_model(model, device, test_loader)
    with open(os.path.join(logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        for character in range(test_prediction.shape[0]):
            if test_prediction[character] == 1:
                print(uppercase_data.test.text[character].upper(), file=predictions_file, end='')
            else:
                print(uppercase_data.test.text[character].lower(), file=predictions_file, end='')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
