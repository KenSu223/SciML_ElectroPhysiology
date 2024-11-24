import torch

file_path = 'darcy_train_16.pt'
checkpoint = torch.load(file_path)

if isinstance(checkpoint, dict):
    print("Attributes and Shapes:")
    for name, tensor in checkpoint.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{name}: {tensor.shape}")
            print(tensor)
        else:
            print(f"{name}: Not a tensor (Type: {type(tensor)})")
else:
    print("The loaded .pt file does not contain a dictionary.")
