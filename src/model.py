import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 30 → 32
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim)  # 32 → 15
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def reconstruction_error(self, x):
        reconstructed = self.forward(x)
        return torch.mean((x - reconstructed) ** 2, dim=1)