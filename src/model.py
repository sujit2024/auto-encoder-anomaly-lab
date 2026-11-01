import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        array = np.asarray(X, dtype=np.float32)
        loader = DataLoader(TensorDataset(torch.from_numpy(array)), batch_size=batch_size, shuffle=False)
        outputs = []

        with torch.no_grad():
            for (batch,) in loader:
                recon = self(batch.to(device)).cpu().numpy()
                outputs.append(recon)

        return np.concatenate(outputs, axis=0) if outputs else np.empty_like(array)


def build_autoencoder(input_dim: int, hidden_dims: list[int] | None = None, latent_dim: int = 8) -> Autoencoder:
    if hidden_dims is None:
        hidden_dims = [64, 32]
    return Autoencoder(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)