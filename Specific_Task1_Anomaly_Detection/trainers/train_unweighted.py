import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ContrastiveModel, Classifier
from data import get_contrastive_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

results_dir = r'C:\Users\Programmer2\Desktop\Specific_Task1_Anomaly detection\results\no_weightdecay_256_30epochs'
os.makedirs(results_dir, exist_ok=True)

def contrastive_loss(z1, z2, label, margin=1.0):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    distance = F.pairwise_distance(z1, z2)
    loss = label * distance.pow(2) + \
           (1 - label) * F.relu(margin - distance).pow(2)
    return loss.mean()

train_loader, val_loader, X_train, y_train, X_test, y_test = \
    get_contrastive_loaders(max_samples=139306, batch_size=64)

model = ContrastiveModel(latent_dim=256, projection_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train_contrastive(model, train_loader, val_loader, optimizer, epochs=30):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}...")

        model.train()
        train_loss = 0
        batch_count = 0

        for x1, x2, labels in train_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = contrastive_loss(z1, z2, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1

            if batch_count % 200 == 0:
                print(f"  Batch {batch_count}, loss: {train_loss/batch_count:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x1, x2, labels in val_loader:
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = contrastive_loss(z1, z2, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

train_losses, val_losses = train_contrastive(
    model, train_loader, val_loader, optimizer, epochs=30
)

torch.save(model.state_dict(), os.path.join(results_dir, 'contrastive_model_no_wd_256.pth'))
print("Model saved!")

for param in model.encoder.parameters():
    param.requires_grad = False

classifier = Classifier(latent_dim=256).to(device)
clf_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
clf_criterion = nn.CrossEntropyLoss()

from torch.utils.data import TensorDataset, DataLoader as TDL

train_clf_loader = TDL(
    TensorDataset(X_train, y_train.long()),
    batch_size=64, shuffle=True
)

def train_classifier(encoder, classifier, loader, optimizer, criterion, epochs=20):
    clf_train_losses = []
    clf_train_accs = []

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                representation, _ = encoder(x)
            out = classifier(representation)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(loader)
        avg_acc  = correct / total
        clf_train_losses.append(avg_loss)
        clf_train_accs.append(avg_acc)

        print(f"Classifier Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} — Acc: {avg_acc:.4f}")

    return clf_train_losses, clf_train_accs

clf_losses, clf_accs = train_classifier(
    model, classifier, train_clf_loader, clf_optimizer, clf_criterion, epochs=20
)

torch.save(classifier.state_dict(), os.path.join(results_dir, 'classifier_no_wd_256.pth'))
print("Classifier saved!")

model.eval()
classifier.eval()

all_probs = []
all_labels = []

test_loader = TDL(
    TensorDataset(X_test, y_test.long()),
    batch_size=64, shuffle=False
)

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        representation, _ = model(x)
        out = classifier(representation)
        probs = torch.softmax(out, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.numpy())

auc = roc_auc_score(all_labels, all_probs)
fpr, tpr, _ = roc_curve(all_labels, all_probs)

print(f"\nTest ROC-AUC: {auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'No Weight Decay 256dim (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve — No Weight Decay 256dim', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'roc_curve_contrastive.png'), dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Contrastive Loss')
plt.title('Contrastive Training Loss Curve — No Weight Decay 256dim')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'contrastive_loss_curve.png'), dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(clf_losses, label='Classifier Train Loss', linewidth=2, color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Classifier Training Loss Curve — No Weight Decay 256dim')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'classifier_loss_curve.png'), dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(clf_accs, label='Classifier Train Accuracy', linewidth=2, color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Classifier Training Accuracy Curve — No Weight Decay 256dim')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'classifier_accuracy_curve.png'), dpi=150, bbox_inches='tight')
plt.show()

with open(os.path.join(results_dir, 'auc_score.txt'), 'w') as f:
    f.write(f'No Weight Decay 256dim 30 Epochs — Test ROC-AUC: {auc:.4f}\n')

print("All results saved!")