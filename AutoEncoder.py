import torch.nn as nn


class Encoder(nn.Sequential):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__(
            # 1x128x128x128
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            # 16x64x64x64
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            # 32x32x32x32
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            # 64x16x16x16
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            # 128x8x8x8
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            # 256x4x4x4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4 * 4, latent_dim)
        )


class Decoder(nn.Sequential):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__(
            nn.Linear(latent_dim, 256 * 4 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4, 4)),
            # 256x4x4x4
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 128x8x8x8
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 64x16x16x16
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 32x32x32x32
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 16x64x64x64
            nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
            # 1x128x128x128
        )


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed