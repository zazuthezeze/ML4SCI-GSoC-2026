import matplotlib.pyplot as plt
import numpy as np

def plot_jet(X, y, index):
    jet = X[index]
    label = "Quark" if y[index] == 1 else "Gluon"
    channel_names = ['ECAL', 'HCAL', 'Tracks']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Jet {index} — {label}', fontsize=14)

    for i in range(3):
        axes[i].imshow(jet[i], cmap='hot')
        axes[i].set_title(channel_names[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()