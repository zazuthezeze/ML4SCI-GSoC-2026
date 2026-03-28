import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset

file_path = r"C:\Users\Programmer2\Desktop\manu\quark-gluon_data-set_n139306.hdf5"

def load_data(batch_size=32, max_samples=20000):
    print("Loading data into RAM...")
    
    with h5py.File(file_path, 'r') as f:
        X = f['X_jets'][:max_samples].astype(np.float32)
        y = f['y'][:max_samples].astype(np.float32)

    print("Transposing...")
    X = np.transpose(X, (0, 3, 1, 2))
    
        # normalize per jet — boost non-zero values
    def normalize_jet(x):
        # x shape: (3, 125, 125)
        for c in range(3):
            channel = x[c]
            max_val = channel.max()
            if max_val > 0:
                x[c] = channel / max_val  # each channel normalized to 0-1 independently
        return x

    print("Normalizing...")
    for i in range(len(X)):
        X[i] = normalize_jet(X[i])

    print("Converting to tensors...")
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    total = len(X_tensor)
    train_end = int(0.8 * total)
    val_end   = int(0.9 * total)

    X_train = X_tensor[:train_end]
    X_val   = X_tensor[train_end:val_end]
    X_test  = X_tensor[val_end:]

    y_train = y_tensor[:train_end]
    y_val   = y_tensor[train_end:val_end]
    y_test  = y_tensor[val_end:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"Training jets:   {len(X_train)}")
    print(f"Validation jets: {len(X_val)}")
    print(f"Test jets:       {len(X_test)}")

    return train_loader, val_loader, test_loader, X_test, y_test

