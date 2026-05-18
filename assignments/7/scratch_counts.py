from morpho_dataset import MorphoDataset
from collections import Counter

morpho = MorphoDataset("czech_cac")
word_counts = Counter(w for sentence in morpho.train.forms.strings for w in sentence)
valid_words = [w for w in morpho.train.forms.word_mapping_vocabulary if word_counts[w] >= 2]

print("Original vocab:", len(morpho.train.forms.word_mapping_vocabulary))
print("Valid vocab (>=2):", len(valid_words))
