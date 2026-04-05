#!/usr/bin/env python3
# file: example_different_model_APIs.py

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
import time

print("Num GPUs Available: ", torch.cuda.device_count())

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_sizes", default="100, 100", type=str, help="Hidden layer sizes.")
parser.add_argument("--model_type", default="sequential", choices=["sequential", "class_API"], help="Model type (sequential, class_API).")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use. Zero value for default.")

def train_model(model, device, train_loader, dev_loader, optimizer, criterion, writer, init_epoch, epochs):

    # Training loop
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
            batch_idx += 1
            # if batch_idx % 100 == 0:
            #     print(f"train loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= num_batches
        train_time = time.time() - start_time
        # print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - train_time: {train_time:.4f} ms')
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        # Validation
        start_time = time.time()
        # Set the evaluation mode flag, however here it is without any effect
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
                # if batch_idx % 100 == 0:
                #     print(f"val loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

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

    # Test
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


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.ModuleList()
        # nn.ModuleDict() or self.add_module(name, module) can be also used.
        # In the latter case, the added modules can be also accessed through
        # parent module attributes created from their names.
        self.relu = nn.ReLU()

        for hidden_layer_size in hidden_layer_sizes:
            hidden_layer_size=int(hidden_layer_size)
            self.fc_layers.append(nn.Linear(input_size, hidden_layer_size))
            input_size = hidden_layer_size

        self.fc_output = nn.Linear(input_size, output_size)
        # self.softmax = nn.Softmax(dim=1) # dim=0 is for batch size, we do not need softmax, because we use nn.CrossEntropyLoss

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.fc_layers:
            x = layer(x)
            x = self.relu(x)
        x = self.fc_output(x)
        # x = self.softmax(x)
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

    args.hidden_layer_sizes = args.hidden_layer_sizes.split(',')
    if args.hidden_layer_sizes == ['']:
        args.hidden_layer_sizes=[]


    # Create logdir name
    logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()
    train_loader = DataLoader(mnist.train, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(mnist.dev, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(mnist.test, batch_size=args.batch_size, shuffle=False)

    for X, y in train_loader:
        print(f"Shape and type of images ([N, C, H, W]): {X.shape}, {X.dtype}")
        print(f"Shape and type of labels: {y.shape}, {y.dtype}")
        break

    # Get cpu, gpu or mps device for training
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Create the model
    if args.model_type=="sequential":
        model = nn.Sequential()
        model.append(nn.Flatten())

        input_size = MNIST.C * MNIST.H * MNIST.W

        for hidden_layer_size in args.hidden_layer_sizes:
            hidden_layer_size=int(hidden_layer_size)
            model.append(nn.Linear(input_size, hidden_layer_size))
            model.append(nn.ReLU())
            input_size = hidden_layer_size

        model.append(nn.Linear(input_size, MNIST.LABELS))
    elif args.model_type=="class_API":
        model = SimpleNN(MNIST.C * MNIST.H * MNIST.W, args.hidden_layer_sizes, MNIST.LABELS)

    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('==================')
    print(f'Total parameters:{pytorch_total_params}')
    print(f'Trainable parameters:{pytorch_trainable_params}')
    print(f'Non-trainable parameters:{pytorch_total_params - pytorch_trainable_params}')
    print('==================')

    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TensorBoard writer initialization
    writer = SummaryWriter(logdir)

    init_epoch=0

    # Training loop
    train_acc, val_acc = train_model(model,
                device,
                train_loader,
                dev_loader,
                optimizer,
                criterion,
                writer,
                init_epoch, args.epochs)
    # Test
    test_acc = eval_model(model,
               device,
               test_loader,
               criterion,
               writer, init_epoch+args.epochs-1)

    print(f"Model accuracies on train/dev/test: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}", flush=True)
    torch.save(model, os.path.join(logdir, "model.pt"))

    writer.flush()
    writer.close()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
