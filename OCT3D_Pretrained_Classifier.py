import torch
import torch.nn as nn


#encoded images as input
class PretrainedClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes, n_demographics):
        super().__init__()
        #self.norm = nn.LayerNorm(latent_dim)
        self.drop = nn.Dropout(p=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + n_demographics, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, demo):
        z = self.drop(self.norm(x))
        return self.classifier(torch.cat([z, demo], dim=1))
