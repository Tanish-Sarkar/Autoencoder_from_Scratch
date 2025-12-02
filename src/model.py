import torch
from torch import nn

class Autoencoder(nn.Module):
     """
    Simple fully-connected autoencoder for tabular data.

    Args:
        input_dim (int): Number of input features.
        latent_dim (int): Size of the bottleneck latent space.
    """
     def __init__(self, input_dim: int, latent_dim: int = 16):
          super().__init__()
          self_encoder = nn.Sequential(
               nn.Linear(input_dim, 128),
               nn.ReLU(),
               nn.Linear(128, 64),
               nn.ReLU(),
               nn.Linear(64, latent_dim)
          )
          self_decoder = nn.Sequential(
               nn.Linear(latent_dim, 128),
               nn.ReLU(),
               nn.Linear(128, 64),
               nn.ReLU(),
               nn.Linear(64, input_dim),
          )
     
     def forward(self, x: torch.Tensor) -> torch.Tensor:
          z = self.encoder(x)
          recon = self.decoder(z)
          return recon
     
def build_autoencoder(input_dim: int, latent_dim: int = 16) -> Autoencoder:
     return Autoencoder(input_dim=input_dim, latent_dim=latent_dim)

if __name__ == "__main__":
     dummy = torch.randn(4,30)
     model = build_autoencoder(input_dim=30, latent_dim=8)
     out = model(dummy)
     print("Input shape:", dummy.shape)
     print("Output shape:", out.shape)
