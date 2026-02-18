import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DenoisingMLP(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, num_layers=4, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encodage du temps
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_layer = nn.Linear(latent_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x_emb = self.input_layer(x)
        h = x_emb
        for layer in self.layers:
            h = h + layer(h + t_emb) # Conditionnement temporel + lien résiduel
        return self.final_layer(h)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, num_steps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.num_steps = num_steps
        
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def p_loss(self, x_0):
        """Calcule la perte d'entraînement"""
        batch_size = x_0.shape[0]
        # 1. Choisir un pas de temps aléatoire
        t = torch.randint(0, self.num_steps, (batch_size,), device=self.device).long()
        
        # 2. Créer du bruit
        noise = torch.randn_like(x_0)
        
        # 3. Bruiter l'image (x_t)
        sqrt_alpha_t = torch.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alpha_t = torch.sqrt(1. - self.alphas_cumprod[t])[:, None]
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        
        # 4. Prédire le bruit
        predicted_noise = self.model(x_t, t)
        
        # 5. Loss MSE
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, n_samples, latent_dim):
        """Génère des échantillons latents"""
        self.model.eval()
        x = torch.randn((n_samples, latent_dim)).to(self.device)
        
        for i in reversed(range(0, self.num_steps)):
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x, t)
            
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
            
        self.model.train()
        return x
