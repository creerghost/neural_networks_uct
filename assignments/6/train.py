import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from cifar10 import CIFAR10

def get_resnet18():
    # Load a ResNet18 architecture without pretrained weights
    model = models.resnet18(weights=None, num_classes=10)
    
    # Adapt the first layer and maxpool for 32x32 CIFAR images instead of 224x224 ImageNet
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def main(args):
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ---------------------------------------------------------
    # 1. Data Augmentation and Normalization
    # ---------------------------------------------------------
    # Mean and std for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
        # Cutout/RandomErasing heavily combats overfitting
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # ---------------------------------------------------------
    # 2. Dataset and DataLoaders
    # ---------------------------------------------------------
    print("Loading CIFAR-10 data...")
    dataset = CIFAR10(transform=train_transform)
    
    # Override transforms for dev and test sets
    dataset.dev.transform = eval_transform
    dataset.test.transform = eval_transform
    
    train_loader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dataset.dev, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # ---------------------------------------------------------
    # 3. Model, Optimizer, Scheduler
    # ---------------------------------------------------------
    model = get_resnet18().to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # SGD with Nesterov momentum and weight decay is standard for ResNets
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    
    # Cosine Annealing smoothly drops the learning rate to 0 by the final epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for inputs, targets in train_loader:
            # cifar10.py yields uint8 targets, ensure they are long for CrossEntropyLoss
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_acc = 100. * correct / total
        train_time = time.time() - start_time
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dev_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        val_acc = 100. * correct / total
        
        # Step the learning rate scheduler
        scheduler.step()
        
        print(f"Epoch: {epoch+1:03d}/{args.epochs:03d} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(dev_loader):.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {train_time:.1f}s")
              
        if val_acc > best_acc:
            print(f"--> New best accuracy! {best_acc:.2f}% -> {val_acc:.2f}%. Saving model...")
            best_acc = val_acc
            # Save entire model to easily load it in 06_code_check.py
            torch.save(model, args.model)
            
    print(f"Training completed. Best Dev Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--model", type=str, default="model.pt", help="Path to save the best model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
