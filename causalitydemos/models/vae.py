from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class ConditionalVAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden_dim=200):
        super().__init__()

        self.encoder_fc1 = nn.Linear(x_dim + y_dim, hidden_dim)
        self.encoder_fc2_mean = nn.Linear(hidden_dim, z_dim)
        self.encoder_fc2_logvar = nn.Linear(hidden_dim, z_dim)
        self.decoder_fc1 = nn.Linear(z_dim + y_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, x_dim)

    def encode(self, x, y):
        """Parametrise q(z|x, y)

        Returns:
            Variational approximation of mean and logvariance of z given x, y
        """
        encoder_input = torch.cat((x, y), dim=-1)
        h1 = F.relu(self.encoder_fc1(encoder_input))
        return self.encoder_fc2_mean(h1), self.encoder_fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        """Predict p(x|z, y)
        """
        decoder_input = torch.cat((z, y), dim=-1)
        h1 = F.relu(self.decoder_fc1(decoder_input))
        return self.decoder_fc2(h1)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def cond_vae_loss_function(recon_x, x, mu, logvar):
    # The log-probability of x | z, y
    decoding_error = torch.sum(-0.5 * (x - recon_x).pow(2))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return decoding_error + kl_div