#!/usr/bin/env python3
# file: gym_cartpole.py
import argparse
import datetime
import os
import re

# uncomment to run on CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

parser = argparse.ArgumentParser()

parser.add_argument("--evaluate_only", default=False, action="store_true", help="Evaluate the given model")
parser.add_argument("--render", default=False, action="store_true", help="Render during evaluation")

parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

parser.add_argument("--model", default="gym_cartpole_model.pt", type=str, help="Output model path.")

# TODO: Add other arguments for the parser if you want.
parser.add_argument("--batch_size", default=7, type=int, help="Batch size.")
parser.add_argument("--epochs", default=150, type=int, help="Number of epochs.")
parser.add_argument("--final_layer", default="softmax", choices=["softmax", "sigmoid"], help="Final layer type.")



def evaluate_model(model, seed=42, episodes=100, render=False, report_per_episode=False, device="cpu"):
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
    """
    import gymnasium as gym

    # Create the environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset(seed=seed)

    model.to(device)
    model.eval()
    # Evaluate the episodes
    total_score = 0
    for episode in range(episodes):
        observation, score, done = env.reset()[0], 0, False
        while not done:
            with torch.no_grad():
                observation = torch.from_numpy(observation).unsqueeze(dim=0)
                observation = observation.to(device)
                prediction = model(observation)[0]
                prediction = prediction.numpy()
            if len(prediction) == 1:
                action = 1 if prediction[0] >= 0.0 else 0
            elif len(prediction) == 2:
                action = np.argmax(prediction)
            else:
                raise ValueError("Unknown model output shape, only 1 or 2 outputs are supported")

            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated

        total_score += score
        if report_per_episode and episode % 10 == 0:
            print("The episode {} finished with score {}.".format(episode + 1, score))
    return total_score / episodes


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

    if not args.evaluate_only:
        
        assert args.epochs is not None, "Select appropriate args.epochs parameter!"
        assert args.batch_size is not None, "Select appropriate args.batch_size parameter!"
        assert args.final_layer is not None, "Select appropriate args.final_layer parameter!"
        
        # Create logdir name
        logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(
                ("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
        ))

        # Get cpu, gpu or mps device for training
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using {device} device")

        # Load the data
        data = np.loadtxt("gym_cartpole_data.txt")
        observations = data[:, :-1].astype(np.float32)
        if args.final_layer=="sigmoid":
            labels = data[:, -1].astype(np.float32)
            labels = labels.reshape((-1,1))
        elif args.final_layer=="softmax":
            labels=np.zeros((len(data),2),dtype=np.float32)
            labels[np.arange(len(labels)), data[:, -1].astype(np.int32)]=1.0

        # observations_dataset = TensorDataset(torch.tensor(observations), torch.tensor(labels))  # create your dataset
        observations_dataset = TensorDataset(torch.from_numpy(observations), torch.from_numpy(labels))  # create your dataset

        train_loader = DataLoader(observations_dataset, batch_size=args.batch_size, shuffle=True)

        # Create the model
        model = nn.Sequential()
        
        input_size = observations.shape[1]
        
        model.append(nn.Linear(input_size, 16))
        model.append(nn.ReLU())
        
        input_size = 16
        # 1. Training dataset is small, 1 hidden layer suits the best
        # 2. Small batch size will add gradient noise, helping the optimizer to generalize better
        # 3. Adam optimizer suited the best for this task to handle the noisy gradient updates
        # 4. Weight decay of 1e-3 was best for this task
        # 5. Number of epochs was increased due to lower learning rate and small batch size
        if args.final_layer=="softmax":
            model.append(nn.Linear(input_size, 2))
        elif args.final_layer == "sigmoid":
            model.append(nn.Linear(input_size, 1))

        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('==================')
        print(f'Total parameters:{pytorch_total_params}')
        print(f'Trainable parameters:{pytorch_trainable_params}')
        print(f'Non-trainable parameters:{pytorch_total_params - pytorch_trainable_params}')
        print('==================')

        model.to(device)


        # Select loss and optimizer.
        if args.final_layer == "softmax":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
            
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)

        # TensorBoard writer initialization
        writer = SummaryWriter(logdir)

        # Training loop

        # Set the training mode flag
        model.train()

        for epoch in range(args.epochs):
            if epoch % 10 == 0:
                print(f'Epoch {epoch + 1}/{args.epochs}:')
            # Training
            start_time = time.time()
            train_loss, train_correct = 0, 0
            num_batches = len(train_loader)
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                train_loss += loss.item()
                if args.final_layer == "sigmoid":
                    train_correct += ((outputs.data>=0).float() == labels).sum().item()
                elif args.final_layer == "softmax":
                    train_correct += (torch.argmax(outputs.data,1) == torch.argmax(labels,1)).sum().item()

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
            writer.flush()

            if epoch % 100 == 0:
                print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - train_time: {train_time:.4f} s')

        writer.close()

        # Save the model, without the optimizer state.
        torch.save(model, args.model)

    # Evaluating, always on cpu here, however you can change it.
    model = torch.load(args.model, map_location="cpu", weights_only=False)

    
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('==================')
    print(f'Total parameters:{pytorch_total_params}')
    print(f'Trainable parameters:{pytorch_trainable_params}')
    print(f'Non-trainable parameters:{pytorch_total_params - pytorch_trainable_params}')
    print('==================')


    score = evaluate_model(model, seed=args.seed, render=args.render, report_per_episode=True)
    print("The average score was {}.".format(score))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
