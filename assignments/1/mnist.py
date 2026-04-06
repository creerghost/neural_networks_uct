import os
import sys
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
# import torch


class MNISTDataset(Dataset):
    def __init__(self, data):
        # self.images = data["images"].astype(np.float32) / 255
        self.images = data["images"]
        # self.labels = data["labels"].astype(np.int64)
        self.labels = data["labels"]

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # return torch.tensor(image), torch.tensor(label)


        # # for common formats makes channel first transofrmation and for common formats scales to range [0.0, 1.0]
        return self.transform(image), label




class MNIST:
    H, W, C = 28, 28, 1
    LABELS = 10

    def __init__(self, dataset="mnist", size={}):
        path = "{}.npz".format(dataset)

        if not os.path.exists(path):
            print('You do not have the dataset!', file=sys.stderr)
            exit(1)

        mnist = np.load(path)
        for dataset in ["train", "dev", "test"]:
            # Remove the key prefix ("train_" in "train_images", "train_labels") and restrict the length by the size parameter if passed
            data = dict((key[len(dataset) + 1:], mnist[key][:size.get(dataset, None)]) for key in mnist if
                        key.startswith(dataset))
            setattr(self, dataset, MNISTDataset(data))
