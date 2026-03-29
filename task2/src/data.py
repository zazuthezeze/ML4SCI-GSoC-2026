import torch
import numpy as np
import h5py
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader

file_path = r"C:\Users\Programmer2\Desktop\task 2\quark-gluon_data-set_n139306.hdf5"

def normalize_jet(jet):
    for c in range(3):
        if jet[c].max() > 0:
            jet[c] = jet[c] / jet[c].max()
    return jet

def image_to_graph(jet_image, label, k=8):
    active_mask = jet_image.sum(axis=0) > 0
    y_coords, x_coords = np.where(active_mask)

    if len(x_coords) < 2:
        return None

    x_norm = x_coords / 124.0
    y_norm = y_coords / 124.0

    ecal   = jet_image[0][y_coords, x_coords]
    hcal   = jet_image[1][y_coords, x_coords]
    tracks = jet_image[2][y_coords, x_coords]

    node_features = np.stack([x_norm, y_norm, ecal, hcal, tracks], axis=1)
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([int(label)], dtype=torch.long)

    pos = x[:, :2].cuda()
    edge_index = pyg_nn.knn_graph(pos, k=k)
    edge_index = edge_index.cpu()

    return Data(x=x, edge_index=edge_index, y=y)


def load_graphs(max_samples=139306):
    print("Loading all jets into RAM...")
    with h5py.File(file_path, 'r') as f:
        X = f['X_jets'][:max_samples].astype(np.float32)
        y = f['y'][:max_samples].astype(np.float32)

    print("Transposing...")
    X = np.transpose(X, (0, 3, 1, 2))

    print("Converting jets to graphs...")
    graphs = []

    for i in range(max_samples):
        if i % 5000 == 0:
            print(f"  Processing jet {i}/{max_samples}...")
        jet = normalize_jet(X[i].copy())
        graph = image_to_graph(jet, y[i])
        if graph is not None:
            graphs.append(graph)

    print(f"Total graphs created: {len(graphs)}")
    return graphs


def get_loaders(max_samples=139306, batch_size=64):
    graphs = load_graphs(max_samples)

    total = len(graphs)
    train_end = int(0.8 * total)
    val_end   = int(0.9 * total)

    train_graphs = graphs[:train_end]
    val_graphs   = graphs[train_end:val_end]
    test_graphs  = graphs[val_end:]

    print(f"Training graphs:   {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Test graphs:       {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader