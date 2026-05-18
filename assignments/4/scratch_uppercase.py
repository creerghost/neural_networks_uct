import torch
import torch.nn as nn
from uppercase_data import UppercaseData
from torch.utils.data import DataLoader

data = UppercaseData(window=20, alphabet_size=60)
print("Train size:", data.train.size)
print("Dev size:", data.dev.size)
print("Alphabet size used:", len(data.train.alphabet))

loader = DataLoader(data.train, batch_size=512, shuffle=True)
for x, y in loader:
    print(x.shape, y.shape)
    break
