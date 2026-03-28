import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model import VAE
from data import load_data, file_path
import h5py
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# load data
train_loader, val_loader, test_loader, X_test, y_test = load_data(batch_size=64, max_samples=139306)

# create model
model = VAE(latent_dim=256).to(device)

# vae loss
def vae_loss(reconstruction, target, mean, logvar):
    # reconstruction loss
    weights = torch.ones_like(target)
    weights[target > 0] = 20.0
    recon_loss = (weights * (reconstruction - target) ** 2).mean()
    
    # kl loss with much smaller weight
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    return recon_loss + 0.0001 * kl_loss

criterion = vae_loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# training loop
def train(model, train_loader, val_loader, optimizer, criterion, epochs=50):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}...")

        model.train()
        train_loss = 0
        batch_count = 0

        for batch_images, _ in train_loader:
            batch_images = batch_images.to(device)
            reconstruction, mean, logvar = model(batch_images)
            loss = criterion(reconstruction, batch_images, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1

            if batch_count % 100 == 0:
                print(f"  Batch {batch_count}, loss so far: {train_loss/batch_count:.6f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_images, _ in val_loader:
                batch_images = batch_images.to(device)
                reconstruction, mean, logvar = model(batch_images)
                loss = criterion(reconstruction, batch_images, mean, logvar)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.6f} — Val Loss: {val_loss:.6f}")

    return train_losses, val_losses

train_losses, val_losses = train(model, train_loader, val_loader, optimizer, criterion, epochs=50)

torch.save(model.state_dict(), 'model2_latent256_no_scheduler.pth')
print("Model 2 weights saved!")




def save_reconstructions(model, model_name):
    model.eval()
    channel_names = ['ECAL', 'HCAL', 'Tracks']
    
    # create folder for this model
    os.makedirs(f'results/{model_name}', exist_ok=True)
    
    # find 3 quarks and 3 gluons from test set
    with h5py.File(file_path, 'r') as f:
        # load a chunk to find quark and gluon indices
        y_all = f['y'][:]
    
    quark_indices = np.where(y_all == 1)[0][:3]
    gluon_indices = np.where(y_all == 0)[0][:3]
    
    for jet_type, indices in [('quark', quark_indices), ('gluon', gluon_indices)]:
        for num, index in enumerate(indices):
            with h5py.File(file_path, 'r') as f:
                x = f['X_jets'][index].astype(np.float32)
                y = f['y'][index]

            x = np.transpose(x, (2, 0, 1))
            # normalize per channel
            for c in range(3):
                if x[c].max() > 0:
                    x[c] = x[c] / x[c].max()

            tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
            with torch.no_grad():
                reconstruction, _, _ = model(tensor)
                reconstructed = reconstruction.squeeze(0).cpu().numpy()

            # create figure — 2 rows, 3 columns
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle(f'{model_name} — {jet_type.capitalize()} Jet {num+1} (Index {index})', 
                        fontsize=14, fontweight='bold')

            for i in range(3):
                # original row
                im1 = axes[0, i].imshow(x[i], cmap='hot', vmin=0, vmax=1)
                axes[0, i].set_title(f'Original {channel_names[i]}', fontsize=12)
                axes[0, i].axis('off')
                plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

                # reconstruction row
                im2 = axes[1, i].imshow(reconstructed[i], cmap='hot', vmin=0, vmax=1)
                axes[1, i].set_title(f'Reconstructed {channel_names[i]}', fontsize=12)
                axes[1, i].axis('off')
                plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)

            axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Reconstructed', fontsize=12, fontweight='bold')

            plt.tight_layout()
            plt.savefig(f'results/{model_name}/{jet_type}_{num+1}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {jet_type}_{num+1}.png")


def save_loss_curve(train_losses, val_losses, model_name):
    os.makedirs(f'results/{model_name}', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} — Training Loss Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}/loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss_curve.png")


def calculate_test_mse(model, test_loader, model_name):
    model.eval()
    total_mse = 0
    count = 0
    
    with torch.no_grad():
        for batch_images, _ in test_loader:
            batch_images = batch_images.to(device)
            reconstruction, _, _ = model(batch_images)
            mse = nn.MSELoss()(reconstruction, batch_images)
            total_mse += mse.item()
            count += 1
    
    avg_mse = total_mse / count
    
    # save as text file
    os.makedirs(f'results/{model_name}', exist_ok=True)
    with open(f'results/{model_name}/mse_score.txt', 'w') as f:
        f.write(f'Model: {model_name}\n')
        f.write(f'Test MSE: {avg_mse:.6f}\n')
    
    print(f"{model_name} — Test MSE: {avg_mse:.6f}")
    return avg_mse


# save everything for model 1
model_name = "model2_latent256_no_scheduler"
save_reconstructions(model, model_name)
save_loss_curve(train_losses, val_losses, model_name)
calculate_test_mse(model, test_loader, model_name)