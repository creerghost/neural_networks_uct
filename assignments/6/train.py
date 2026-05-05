import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from cifar10 import CIFAR10
from typing import Any


def get_resnet18() -> Any:
    """
    Load ResNet18 and adapt for 32x32 CIFAR images due to ResNet18's original
    design for ImageNet, which consists of 224x224 pixels.

    Replacing first convolutional layer with a smaller one and deleting the
    first MaxPool layer we can achieve the compatibility of this model
    with CIFAR dataset.
    """
    base_model = models.resnet18(weights=None, num_classes=10)
    base_model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    base_model.maxpool = nn.Identity()

    # Embed data normalization natively into the model
    # so it requires no external transforms
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    norm_layer = transforms.Normalize(mean=mean, std=std)

    model = nn.Sequential(
        norm_layer,
        base_model
    )
    return model


def main(args) -> None:
    """
    Main training loop for CIFAR-10 classification.

    1. Due to massive amount of ResNet18's parameters, the images were mutated
    with mathematical transformations (with torchvision's transforms methods).

        RandomCrop will shift the image slightly off-center.

        RandomHorizontalFlip will mirror the image 50 % of the time.

        RandomErasing will randomly place a black box over a chunk of the
        image. This will prevent the network from relying on a single pixel
        and forces it to learn the whole shape of the object.

    2. SGD was used with high momentum and weight decay.

    3. As a learning rate (LR) scheduler, the cosine annealing was used. It
    starts the LR around 0.1. As epochs progresses, it smoothly curves the LR
    downwards following a cosine wave until it hits exactly 0.0 on last epoch.

    4. Standard PyTorch training loop was implemented.
    """
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # source: https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data/69699979#69699979 # noqa

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(
            p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3),
            value=0, inplace=False
        )
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor()])

    print("Loading CIFAR-10 data...")
    dataset = CIFAR10(transform=train_transform)

    dataset.dev.transform = eval_transform
    dataset.test.transform = eval_transform

    train_loader = DataLoader(
        dataset.train, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    dev_loader = DataLoader(
        dataset.dev, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model = get_resnet18().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=5e-4, nesterov=True
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_acc = 0.0

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for inputs, targets in train_loader:
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

        scheduler.step()

        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(dev_loader)
        print(
            f"Epoch: {epoch+1:03d}/{args.epochs:03d} | "
            f"Train Loss: {t_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {v_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {train_time:.1f}s"
        )

        if val_acc > best_acc:
            print(
                f"New best accuracy! {best_acc:.2f}% -> "
                f"{val_acc:.2f}%. Saving model..."
            )
            best_acc = val_acc
            torch.save(model, args.model)

    print(f"Training completed. Best Dev Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Initial learning rate"
    )
    parser.add_argument(
        "--model", type=str, default="model.pt", help="Path to save model"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()
    main(args)
