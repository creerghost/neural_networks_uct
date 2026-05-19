import torch
weights = torch.load('model_weights.tar', map_location='cpu')
for k, v in weights.items():
    print(k, v.shape)
