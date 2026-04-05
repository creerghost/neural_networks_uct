import os
import sys
import zipfile
import torch
from torch.utils.data import Dataset

import numpy as np

# Loads the Uppercase data.
# - The data consists of three Datasets
#   - train
#   - dev
#   - test [all in lowercase]
# - When loading, maximum number of alphabet characters can be specified,
#   in which case that many most frequent characters will be used, and all
#   other will be remapped to "<unk>".
# - Features are generated using a sliding window of given size,
#   i.e., for a character, we include left `window` characters, the character
#   itself and right `window` characters, `2 * window + 1` in total.
# - Each dataset (train/dev/test) has the following members:
#   - size: the length of the text
#   - windows: input examples with shape [size, 2 * window_size + 1],
#       corresponding to indices of input lowercased characters
#   -labels: input labels with shape [size], each a 0/1 value whether
#       the corresponding input in `windows` is lowercased/uppercased
#   - text: the original text (of course lowercased in case of the test set)
#   - alphabet: an alphabet used by `windows`
#   - dataset: a PyTorch torch.utils.data.Dataset producing as examples 
#       "windows" and "labels"

class UppercaseDataset(Dataset):
    def __init__(self, data, window, alphabet, seed=42):
        self.text = data
        self.size = len(self.text)
        self.window = window

        # Create alphabet_map
        alphabet_map = {"<pad>": 0, "<unk>": 1}
        if alphabet and not isinstance(alphabet, int):
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index
        else:
            if not alphabet or alphabet<0:
                alphabet = np.inf
            # Find most frequent characters
            freqs = {}
            for char in self.text.lower():
                freqs[char] = freqs.get(char, 0) + 1
            init_len_alphabet = len(alphabet_map)

            most_frequent = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
            for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                alphabet_map[char] = i
                if len(alphabet_map) >= alphabet + init_len_alphabet:
                    break

        # Remap lowercased input characters using the alphabet_map
        lcletters = np.zeros(self.size + 2 * window, np.int16)
        for i in range(self.size):
            char = self.text[i].lower()
            if char not in alphabet_map:
                char = "<unk>"
            lcletters[i + window] = alphabet_map[char]

        # Generate input batches
        windows = np.zeros([self.size, 2 * window + 1], np.int16)
        labels = np.zeros(self.size, np.uint8)
        for i in range(self.size):
            windows[i] = lcletters[i:i + 2 * window + 1]
            labels[i] = self.text[i].isupper()
        self.windows = windows
        self.labels = labels

        # Compute alphabet
        self.alphabet = [None] * len(alphabet_map)
        for key, value in alphabet_map.items():
            self.alphabet[value] = key


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.labels[idx]
        return torch.from_numpy(window), label



class UppercaseData:
    LABELS = 2

    def __init__(self, window, alphabet_size):
        path = 'uppercase_data.zip'
        if not os.path.exists(path):
            print('You do not have the dataset!', file=sys.stderr)
            exit(1)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    data = dataset_file.read().decode("utf-8")
                setattr(self, dataset, UppercaseDataset(
                    data,
                    window,
                    alphabet=alphabet_size if dataset == "train" else self.train.alphabet,
                ))

    # Evaluation infrastructure.
    @staticmethod
    def evaluate_text(gold, prediction):
        if len(prediction) < len(gold):
            raise RuntimeError("The predictions are shorter than gold data: {} vs {}.".format(
                len(prediction), len(gold)))

        correct = 0
        for i in range(len(gold)):
            # Note that just the lower() condition is not enough, for example
            # u03c2 and u03c3 have both u03c2 as an uppercase character.
            if prediction[i].lower() != gold[i].lower() and prediction[i].upper() != gold[i].upper():
                raise RuntimeError("The predictions and gold data differ on position {}: {} vs {}.".format(
                    i, repr(prediction[i:i + 20].lower()), repr(gold[i:i + 20].lower())))

            correct += gold[i] == prediction[i]
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_dataset(gold_dataset, prediction_file):
        gold = gold_dataset.text
        prediction = prediction_file.read()
        return UppercaseData.evaluate_text(gold, prediction)

    @staticmethod
    def evaluate_file(gold_file, prediction_file):
        gold = gold_file.read()
        prediction = prediction_file.read()
        return UppercaseData.evaluate_text(gold, prediction)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--evaluate_against_file", default=None, type=str, help="Gold file to evaluate")
    parser.add_argument("--evaluate_against_dataset", default=None, choices=["dev", "train"], help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate_against_file:
        with open(args.prediction_file, "r", encoding="utf-8-sig") as prediction_file:
            with open(args.evaluate_against_file, "r", encoding="utf-8-sig") as gold_file:
                accuracy = UppercaseData.evaluate_file(gold_file, prediction_file)
        print("Uppercase accuracy: {:.2f}%".format(accuracy))
    elif args.evaluate_against_dataset:
        with open(args.prediction_file, "r", encoding="utf-8-sig") as prediction_file:
            accuracy = UppercaseData.evaluate_dataset(getattr(UppercaseData(0,0), args.evaluate_against_dataset), prediction_file)
        print("Uppercase accuracy: {:.2f}%".format(accuracy))
