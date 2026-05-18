import torch
import uppercase
model = uppercase.SimpleNN(41, 64)
state_dict = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state_dict)
torch.save(model, "model.pt")
print("Successfully converted model.pt to full model object.")
