import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(256 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ProjectionHead(nn.Module):
    """
    Small network on top of encoder that maps to contrastive space.
    Used during contrastive training only — discarded for classification.
    """
    def __init__(self, latent_dim=128, projection_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim)
        )

    def forward(self, x):
        return self.fc(x)


class ContrastiveModel(nn.Module):
    def __init__(self, latent_dim=128, projection_dim=64):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.projector = ProjectionHead(latent_dim, projection_dim)

    def forward(self, x):
        representation = self.encoder(x)
        projection = self.projector(representation)
        return representation, projection


class Classifier(nn.Module):
    """
    Simple classifier added on top of frozen encoder for evaluation.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)