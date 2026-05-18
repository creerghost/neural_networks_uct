from morpho_dataset import MorphoDataset
morpho = MorphoDataset("czech_cac")
print("Train sentences:", len(morpho.train))
print("Forms vocabulary:", len(morpho.train.forms.word_mapping_vocabulary))
print("Tags vocabulary:", len(morpho.train.tags.word_mapping_vocabulary))
