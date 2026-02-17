import torch
import torch.nn as nn
import torch.nn.functional as F

class Tokenizer(nn.Module):
    def __init__(self, num_numerical: int, cat_cardinalities: list[int], d: int):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.num_categorical = len(cat_cardinalities)
        self.M = self.num_numerical + self.num_categorical
        self.d = d

        self.num_proj = nn.ModuleList([nn.Linear(1, d) for _ in range(num_numerical)])
        self.cat_emb  = nn.ModuleList([nn.Embedding(card, d) for card in cat_cardinalities])

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = []
        for j in range(self.num_numerical):
            tokens.append(self.num_proj[j](x_num[:, j:j+1]))   
        for j in range(self.num_categorical):
            tokens.append(self.cat_emb[j](x_cat[:, j]))        
        return torch.stack(tokens, dim=1)  

class BetaVAEEncoder(nn.Module):
    def __init__(self, M: int, d: int, n_layers=4, n_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.M, self.d = M, d
        self.pos_emb = nn.Embedding(M, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads,
            dim_feedforward=ff_mult*d, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.mu = nn.Linear(d, d)
        self.logvar = nn.Linear(d, d)

    def forward(self, T):
        B, M, d = T.shape
        pos = torch.arange(M, device=T.device).unsqueeze(0).expand(B, M)
        H = self.enc(T + self.pos_emb(pos))
        mu = self.mu(H)
        logvar = self.logvar(H)
        return mu, logvar  

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class BetaVAEDecoder(nn.Module):
    def __init__(self, M: int, d: int, n_layers=4, n_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.M, self.d = M, d
        self.pos_emb = nn.Embedding(M, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads,
            dim_feedforward=ff_mult*d, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.dec = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, Z):
        # Z: (B,M,d)
        B, M, _ = Z.shape
        pos = torch.arange(M, device=Z.device).unsqueeze(0).expand(B, M)
        H = self.dec(Z + self.pos_emb(pos))
        return self.out_norm(H)  

class BetaVAE(nn.Module):
    def __init__(self, num_numerical: int, cat_cardinalities: list[int], d: int, beta: float = 1.0):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.num_categorical = len(cat_cardinalities)
        self.M = num_numerical + self.num_categorical
        self.d = d
        self.beta = beta

        self.tokenizer = Tokenizer(num_numerical, cat_cardinalities, d)
        self.encoder = BetaVAEEncoder(self.M, d)
        self.decoder = BetaVAEDecoder(self.M, d)


        self.num_head = nn.Linear(d, 1) 
        self.cat_heads = nn.ModuleList([nn.Linear(d, card) for card in cat_cardinalities])

    def forward(self, x_num, x_cat):
        T = self.tokenizer(x_num, x_cat)          
        mu, logvar = self.encoder(T)              
        Z = reparameterize(mu, logvar)            
        T_hat = self.decoder(Z)                   

        
        num_tokens = T_hat[:, :self.num_numerical, :]              
        cat_tokens = T_hat[:, self.num_numerical:, :]              

        x_num_hat = self.num_head(num_tokens).squeeze(-1)         
        x_cat_logits = [head(cat_tokens[:, j, :]) for j, head in enumerate(self.cat_heads)]
        return x_num_hat, x_cat_logits, mu, logvar

    def loss(self, x_num, x_cat, x_num_hat, x_cat_logits, mu, logvar):
        rec_num = F.mse_loss(x_num_hat, x_num, reduction="mean")
        rec_cat = 0.0
        for j, logits in enumerate(x_cat_logits):
            rec_cat = rec_cat + F.cross_entropy(logits, x_cat[:, j], reduction="mean")
        rec = rec_num + rec_cat


        kl = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=(1,2)))
        return rec + self.beta * kl, {"rec": rec.item(), "kl": kl.item()}