from scipy.io import loadmat

file_path = '/home/tianyi/SML/EP-PINNs/data_files/Aliev_Panfilov_Model/_2D/data_2d_spiral.mat'
data = loadmat(file_path)
print("Attributes and Shapes:")
for name, attribute in data.items():
    if not name.startswith('__'):
        if hasattr(attribute, 'shape'):
            print(f"{name}: {attribute.shape}")
        else:
            print(f"{name}: No shape attribute (Type: {type(attribute)})")
