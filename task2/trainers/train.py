import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import GNNClassifier
from data import get_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# load data
train_loader, val_loader, test_loader = get_loaders(max_samples=139306, batch_size=64)

# setup model
model = GNNClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train(model, train_loader, val_loader, optimizer, criterion, epochs=50):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}...")

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y.squeeze()).sum().item()
            total += batch.y.size(0)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y.squeeze())
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y.squeeze()).sum().item()
                val_total += batch.y.size(0)

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        train_acc   = correct / total
        val_acc     = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f} — Train Acc: {train_acc:.4f} — Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


train_losses, val_losses, train_accs, val_accs = train(
    model, train_loader, val_loader, optimizer, criterion, epochs=50
)

torch.save(model.state_dict(), 'gnn_model.pth')
print("GNN model saved!")

# ROC AUC
model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.softmax(out, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch.y.squeeze().cpu().numpy())

auc = roc_auc_score(all_labels, all_probs)
fpr, tpr, _ = roc_curve(all_labels, all_probs)

print(f"\nTest ROC-AUC: {auc:.4f}")

# save results
os.makedirs('results', exist_ok=True)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'GNN (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve — GNN Jet Classifier', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curve_gnn.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GNN Training Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/gnn_loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accs, label='Train Accuracy', linewidth=2)
plt.plot(val_accs, label='Val Accuracy', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('GNN Training Accuracy Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/gnn_accuracy_curve.png', dpi=150, bbox_inches='tight')
plt.show()

with open('results/auc_score.txt', 'w') as f:
    f.write(f'Test ROC-AUC: {auc:.4f}\n')

print("All results saved!")