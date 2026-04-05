import os
import sys
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
# from torchvision.transforms import v2
# import torch


class CIFAR10Dataset(Dataset):
    def __init__(self, data, transform=None):

        self.images = data["images"]
        self.labels = data["labels"].ravel()

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # # for common formats makes channel first transformation and for common formats scales to range [0.0, 1.0]
        return self.transform(image), label

class CIFAR10:
    H, W, C = 32, 32, 3
    LABELS = 10
    LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __init__(self, dataset="cifar10_with_reduced_test_set", sizes={}, transform=None):
        path = "{}.npz".format(dataset)

        if not os.path.exists(path):
            print('You do not have the dataset!', file=sys.stderr)
            sys.exit(-1)

        cifar10 = np.load(path)
        for dataset in ["train", "dev", "test"]:
            # Remove the key prefix ("train_" in "train_images", "train_labels") and restrict the length by the size parameter if passed
            data = dict((key[len(dataset) + 1:], cifar10[key][:sizes.get(dataset, None)]) for key in cifar10 if
                        key.startswith(dataset))
            if dataset=="train":
                setattr(self, dataset, CIFAR10Dataset(data, transform))
            else:
                setattr(self, dataset, CIFAR10Dataset(data))

        # Evaluation infrastructure.
        @staticmethod
        def evaluate(gold_dataset, predictions):
            gold = gold_dataset.data["labels"]

            if len(predictions) != len(gold):
                raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                    len(predictions), len(gold)))

            correct = sum(bool(gold[i] == predictions[i]) for i in range(len(gold)))
            return 100 * correct / len(gold)

        @staticmethod
        def evaluate_file(gold_dataset, predictions_file):
            predictions = [int(line) for line in predictions_file]
            return CIFAR10.evaluate(gold_dataset, predictions)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = CIFAR10.evaluate_file(getattr(CIFAR10(), args.dataset), predictions_file)
        print("CIFAR10 accuracy: {:.2f}%".format(accuracy))

