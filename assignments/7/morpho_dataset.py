import os
import sys
import zipfile

from torch.utils.data import Dataset


# Loads a morphological dataset in a vertical format.
# - The morphological dataset consists of three Datasets
#   - train
#   - dev
#   - test
# - Each Dataset is composed of
#   - size: a number of sentences in the dataset,
#   - get_dataset() property constructing a `torch.utils.data.Dataset` with "form", "lemma", "tag".
#   - forms, lemmas, tags: `Strings_With_Vocabulary` objects containing the following fields:
#       - strings: a Python list containing input sentences, each being
#           a list of strings (forms/lemmas/tags).
#       - word_mapping_vocabulary: a Python list containing all words in input sentences.
#           It is constructed on the train set and is
#           shared by the dev and test sets.
#       - char_mapping_vocabulary: a Python list containing all characters in input sentences.
#           It is constructed on the train set and is
#           shared by the dev and test sets.

class MorphoDataset:
    PAD = 0
    UNK = 1
    BOW = 2
    EOW = 3
    class StringsWithVocabulary:

        word_mapping_vocabulary = None
        char_mapping_vocabulary = None
        def __init__(self):
            self.strings = []

        def create_vocabularies(self, train_data_strings_with_vocabulary = None, add_bow_eow = False):
            if train_data_strings_with_vocabulary:
                # just keep the originals from train
                self.word_mapping_vocabulary = train_data_strings_with_vocabulary.word_mapping_vocabulary
                self.char_mapping_vocabulary = train_data_strings_with_vocabulary.char_mapping_vocabulary
            else:
                self.word_mapping_vocabulary = sorted(set(string for sentence in self.strings for string in sentence))

                additional_characters = []
                if add_bow_eow:
                    additional_characters.extend(["[BOW]", "[EOW]"])
                self.char_mapping_vocabulary = additional_characters + sorted(set(
                        char for sentence in self.strings for string in sentence for char in string))

    class TorchDataset(Dataset):
        def __init__(self, data_file, train_dataset = None,
                     max_sentences = None, add_bow_eow = False):
            # Create factors
            self._factors = (MorphoDataset.StringsWithVocabulary(),
                             MorphoDataset.StringsWithVocabulary(),
                             MorphoDataset.StringsWithVocabulary())

            # Load the data
            self._size = 0
            in_sentence = False
            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                if line:
                    if not in_sentence:
                        for factor in self._factors:
                            factor.strings.append([])
                        self._size += 1

                    columns = line.split("\t")
                    assert len(columns) == len(self._factors)
                    for column, factor in zip(columns, self._factors):
                        factor.strings[-1].append(column)

                    in_sentence = True
                else:
                    in_sentence = False
                    if max_sentences is not None and self._size >= max_sentences:
                        break

            # Create vocabularies
            for i, factor in enumerate(self._factors):
                factor.create_vocabularies(train_dataset._factors[i] if train_dataset else None, add_bow_eow)

        @property
        def forms(self):
            return self._factors[0]

        @property
        def lemmas(self):
            return self._factors[1]

        @property
        def tags(self):
            return self._factors[2]

        def __len__(self):
            return self._size

        def __getitem__(self, index):
            return {"forms": self.forms.strings[index],
                    "lemmas": self.lemmas.strings[index],
                    "tags": self.tags.strings[index]}

        def with_transform(self, transform):
            return MorphoDataset.TransformedTorchDataset(self, transform)

    class TransformedTorchDataset(Dataset):
        def __init__(self, dataset, transform):
            self._dataset = dataset
            self._transform = transform

        def __len__(self):
            return len(self._dataset)

        def __getitem__(self, index):
            item = self._dataset[index]
            return self._transform(*item) if isinstance(item, tuple) else self._transform(item)

        def with_transform(self, transform):
            return MorphoDataset.TransformedTorchDataset(self, transform)

    def __init__(self, dataset_name, max_sentences=None, add_bow_eow=False):

        path = "{}.zip".format(dataset_name)
        if not os.path.exists(path):
            print("You do not have dataset file {}!".format(path), file=sys.stderr)
            sys.exit(-1)

        with zipfile.ZipFile(path, "r") as zip_file:
            dataset = "train"
            with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                setattr(self, dataset, self.TorchDataset(dataset_file,
                                                    None,
                                                    max_sentences=max_sentences, add_bow_eow=add_bow_eow))

            for dataset in ["dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.TorchDataset(dataset_file,
                                                        self.train,
                                                        max_sentences=max_sentences, add_bow_eow=add_bow_eow))


    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset, predictions):
        gold_sentences = gold_dataset.strings

        predicted_sentences, in_sentence = [], False
        for line in predictions:
            line = line.rstrip("\n")
            if not line:
                in_sentence = False
            else:
                if not in_sentence:
                    predicted_sentences.append([])
                    in_sentence = True
                predicted_sentences[-1].append(line)

        if len(predicted_sentences) != len(gold_sentences):
            raise RuntimeError("The predictions contain different number of words than gold data: {} vs {}".format(
                len(predicted_sentences), len(gold_sentences)))

        correct, total = 0, 0
        for i, (predicted_sentence, gold_sentence) in enumerate(zip(predicted_sentences, gold_sentences)):
            if len(predicted_sentence) != len(gold_sentence):
                raise RuntimeError("Predicted sentence {} has different number of words than gold: {} vs {}".format(
                    i + 1, len(predicted_sentence), len(gold_sentence)))
            correct += sum(predicted == gold for predicted, gold in zip(predicted_sentence, gold_sentence))
            total += len(predicted_sentence)

        return 100 * correct / total

    @staticmethod
    def evaluate_file(gold_dataset, predictions_file):
        predictions = predictions_file.readlines()
        return MorphoDataset.evaluate(gold_dataset, predictions)

