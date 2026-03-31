import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

file_path = r"C:\Users\Programmer2\Desktop\Specific_Task1_Anomaly detection\quark-gluon_data-set_n139306.hdf5"

def normalize_jet(jet):
    for c in range(3):
        if jet[c].max() > 0:
            jet[c] = jet[c] / jet[c].max()
    return jet


class JetDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ContrastiveDataset(Dataset):
    """
    Returns pairs of jets and whether they are the same type or not.
    For each jet randomly picks another jet of the same type (positive)
    or different type (negative).
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

        # precompute indices for each class
        self.quark_indices = np.where(y == 1)[0]
        self.gluon_indices = np.where(y == 0)[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x1 = self.X[idx]
        y1 = self.y[idx]

        # 50% chance of positive pair, 50% negative pair
        if np.random.random() > 0.5:
            # positive pair — same type
            if y1 == 1:
                idx2 = np.random.choice(self.quark_indices)
            else:
                idx2 = np.random.choice(self.gluon_indices)
            label = 1  # similar
        else:
            # negative pair — different type
            if y1 == 1:
                idx2 = np.random.choice(self.gluon_indices)
            else:
                idx2 = np.random.choice(self.quark_indices)
            label = 0  # different

        x2 = self.X[idx2]

        return x1, x2, torch.tensor(label, dtype=torch.float)


def load_data(max_samples=139306):
    print("Loading data into RAM...")
    with h5py.File(file_path, 'r') as f:
        X = f['X_jets'][:max_samples].astype(np.float32)
        y = f['y'][:max_samples].astype(np.float32)

    print("Transposing...")
    X = np.transpose(X, (0, 3, 1, 2))

    print("Normalizing...")
    for i in range(len(X)):
        X[i] = normalize_jet(X[i])

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    # split 80/10/10
    total = len(X)
    train_end = int(0.8 * total)
    val_end   = int(0.9 * total)

    X_train, y_train = X[:train_end],      y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],         y[val_end:]

    print(f"Training:   {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test:       {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_contrastive_loaders(max_samples=139306, batch_size=64):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(max_samples)

    train_dataset = ContrastiveDataset(X_train.numpy(), y_train.numpy())
    val_dataset   = ContrastiveDataset(X_val.numpy(),   y_val.numpy())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, y_train, X_test, y_test