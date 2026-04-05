#!/usr/bin/env python3
# file: mnist_cnn.py

import argparse
import datetime
import re
import os
import sys

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
parser.add_argument("--cnn", default="", type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
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

# TODO: Add CNN layers specified by `args.cnn`, which contains
# a comma-separated list of the following layers:
# - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
#   activation and specified number of filters, kernel size, stride and padding.
# - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
#   In detail, start with a convolutional layer without bias and activation,
#   then add a batch normalization layer, and finally the ReLU activation.
# - `M-pool_size-stride-padding`: Add max pooling with specified size, stride and padding.
# - `R-[layers]`: Add a residual connection. The `layers` contain a specification
#   of at least one convolutional layer (but not a recursive residual connection `R`).
#   The input to the `R` layer should be processed sequentially by `layers`, and the
#   produced output (after the ReLU nonlinearity of the last layer) should be added
#   to the input (of this `R` layer).
# - `F`: Flatten inputs. Must appear exactly once in the architecture.
# - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
# - `D-dropout_rate`: Apply dropout with the given dropout rate.
# You can assume the resulting network is valid; it is fine to crash if it is not.
#


class SimpleNN(nn.Module):
    def add_layer(self, layer_string):
        params = layer_string.split("-")

        if (params[0]=="F"):
            #flatten layer
            self.network_layers.append(nn.Flatten())
            self.residual_ops.append("")
        elif (params[0]=="H"):
            #fully connected layer
            self.network_layers.append(nn.LazyLinear(int(params[1])))
            self.residual_ops.append("")

            self.network_layers.append(nn.ReLU())
            self.residual_ops.append("")
        elif (params[0]=="D"):
            #dropout layer
            self.network_layers.append(nn.Dropout(float(params[1])))
            self.residual_ops.append("")
        elif (params[0]=="M"):
            #max poolig layer
            kernel_size = int(params[1])
            stride = int(params[2])
            padding = int(params[3])
            self.network_layers.append(nn.MaxPool2d(kernel_size, stride, padding))
            self.residual_ops.append("")
        elif (params[0]=="C"):
            # convolutional layer
            out_channels = int(params[1])
            kernel_size = int(params[2])
            stride = int(params[3])
            if params[4]!="valid" and params[4]!="same":
                padding = int(params[4])
            # elif params[4]=="same" and int(params[3])>1:
            #     padding=int(kernel_size / 2)
            else:
                padding = params[4]
            self.network_layers.append(nn.LazyConv2d(out_channels,kernel_size, stride, padding))
            self.residual_ops.append("")
            self.network_layers.append(nn.ReLU())
            self.residual_ops.append("")
        elif (params[0] == "CB"):
            # convolutional layer with batch normalization
            out_channels = int(params[1])
            kernel_size = int(params[2])
            stride = int(params[3])
            if params[4] != "valid" and params[4] != "same":
                padding = int(params[4])
            else:
                padding = params[4]
            self.network_layers.append(nn.LazyConv2d(out_channels, kernel_size, stride, padding))
            self.residual_ops.append("")

            self.network_layers.append(nn.LazyBatchNorm2d())
            self.residual_ops.append("")

            self.network_layers.append(nn.ReLU())
            self.residual_ops.append("")
        elif (params[0]=="R"):
            result = re.search("R-\\[(.*)\\]", layer_string)
            skipped_layers_strings = result.group(1)

            skipped_layers = skipped_layers_strings.split("$")


            # start_index = len(self.network_layers_resid_op)
            for i, layer in enumerate(skipped_layers):
                print("Adding layer with residual connection " + str(i) + ":" + layer)
                self.add_layer(layer)
                if (i==0):
                    self.residual_ops[-1] +="-"

            #self.network_layers_resid_op[start_index] +="-"
            self.residual_ops[-1] += "+"

        else:
            print("Unknown layer:" + layer_string, flush=True)
            sys.exit(-1)

    def __init__(self, network_architecture_string, output_size):
        super().__init__()
        
        self.network_layers = nn.ModuleList()
        self.residual_ops = list()

        #self.layers_definition_string = network_architecture_string

        replaced_char_list = list(network_architecture_string)
        flag = False
        for i in range(len(network_architecture_string)):
            if network_architecture_string[i] == "[":
                flag = True
            elif network_architecture_string[i] == "]":
                flag = False
            elif flag and network_architecture_string[i] == ",":
                 replaced_char_list[i] = "$"

        network_architecture_string = "".join(replaced_char_list)

        layers = network_architecture_string.split(",")

        for i, layer in enumerate(layers):
            print("Adding layer " + str(i) + ":" + layer)
            self.add_layer(layer)


        self.fc_output = nn.LazyLinear(output_size)
        # self.softmax = nn.Softmax(dim=1) # dim=0 is for batch size, we do not need softmax, because we use nn.CrossEntropyLoss

    def forward(self, x):
 
        for i, layer in enumerate(self.network_layers):
            if "-" in self.residual_ops[i]:
                x_saved = x

            x = layer(x)

            if "+" in self.residual_ops[i]:
                x = x + x_saved
        #for layer, op in zip(self.network_layers, self.residual_ops):
        #    if "-" in op:
        #        x_saved = x
        #    x = layer(x)
        #    if "+" in op:
        #        x = x + x_saved



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


    # Create logdir name
    logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
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

    # Create the model.
    model = SimpleNN(args.cnn, MNIST.LABELS)

    # TODO: Initialize lazy layers.

    X, y = next(iter(train_loader))
    model.eval()
    outputs = model(X)



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
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

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

    writer.flush()
    writer.close()
    # Save the model to the logdir.
    torch.save(model, os.path.join(logdir, "model.pt"))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
