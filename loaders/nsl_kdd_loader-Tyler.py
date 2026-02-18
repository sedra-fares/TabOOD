"""End-to-end NSL-KDD experiment: load -> one-hot encode -> tokenize -> beta-VAE train."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files",
    "num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]
path_train = "data/nsl_kdd/KDDTrain+.txt"
path_test = "data/nsl_kdd/KDDTest+.txt"
def load_nsl_kdd(path_train, path_test):
    train_df = pd.read_csv(path_train, names=COLUMNS)
    test_df = pd.read_csv(path_test, names=COLUMNS)

    train_df["label"] = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
    test_df["label"] = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)

    train_df.drop(columns=["difficulty"], inplace=True)
    test_df.drop(columns=["difficulty"], inplace=True)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    return X_train, y_train, X_test, y_test

data = load_nsl_kdd(path_train, path_test)
print("Train shape:", data[0].shape, data[1].shape)
print("Test shape:", data[2].shape, data[3].shape)
X = data[0]
y = data[1]

# Identify numeric and categorical feature names
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocess: scale numeric and one-hot encode categoricals to a dense matrix
num_transformer = Pipeline([
    ('variance', VarianceThreshold(threshold=1e-8)),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'), cat_cols)
    ],
    remainder='drop'
)

full_pipe = Pipeline([
    ('preprocessor', preprocessor)
])

X_ready = full_pipe.fit_transform(X)
print("Preprocessed shape (numeric + one-hot cat):", X_ready.shape)

# All features are now numeric; treat them as numeric tokens
x_num = torch.tensor(X_ready, dtype=torch.float32)
print("Tensor for tokenizer (numeric-only) shape:", x_num.shape)

# No categorical indices are needed after one-hot; keep placeholder
x_cat = None


class FeatureTokenizer(nn.Module):
    """Tokenize dense numeric features (after one-hot) into per-feature tokens."""

    def __init__(self, n_features: int, d: int):
        super().__init__()
        self.num_projs = nn.ModuleList([nn.Linear(1, d) for _ in range(n_features)])

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None):
        # x_num: (B, n_features)
        num_tokens = [proj(x_num[:, i:i+1]) for i, proj in enumerate(self.num_projs)]
        tokens = torch.stack(num_tokens, dim=1)  # (B, M, d)
        return tokens


class FeatureDetokenizer(nn.Module):
    """Invert tokens back to numeric features (categoricals already one-hot)."""

    def __init__(self, tokenizer):
        super().__init__()
        self.n_num = len(tokenizer.num_projs)
        self.d = tokenizer.num_projs[0].out_features
        self.num_recons = nn.ModuleList([nn.Linear(self.d, 1) for _ in range(self.n_num)])

    def forward(self, tokens: torch.Tensor):
        # tokens: (B, M, d)
        num_tokens = tokens  # all tokens are numeric in this flow
        recon_nums = [self.num_recons[i](num_tokens[:, i, :]) for i in range(self.n_num)]
        x_num_recon = torch.cat(recon_nums, dim=1)
        return x_num_recon

d = 32  # token embedding dimension
tokenizer = FeatureTokenizer(n_features=x_num.shape[1], d=d)
detokenizer = FeatureDetokenizer(tokenizer)

tokens = tokenizer(x_num)
print("Token shape:", tokens.shape)

x_num_recon = detokenizer(tokens)
print("Reconstructed numeric shape:", x_num_recon.shape)


class TransformerBetaVAE(nn.Module):
    """Transformer-based beta-VAE over tokenized one-hot numeric features."""

    def __init__(self, tokenizer, detokenizer, d_model=64, n_heads=4, n_layers=2, latent_dim=16, beta=4.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.beta = beta

        self.M = len(tokenizer.num_projs)  # number of tokens per sample
        self.d_model = d_model
        self.latent_dim = latent_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

        self.z_to_tokens = nn.Linear(latent_dim, self.M * d_model)

        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

    def encode(self, x_num):
        tokens = self.tokenizer(x_num)             # (B, M, d_model)
        h = self.encoder(tokens)                   # (B, M, d_model)
        h_pool = h.mean(dim=1)                     # (B, d_model)
        mu = self.to_mu(h_pool)                    # (B, latent_dim)
        logvar = self.to_logvar(h_pool)            # (B, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B = z.size(0)
        tokens_seed = self.z_to_tokens(z).view(B, self.M, self.d_model)
        tokens_dec = self.decoder(tokens_seed)
        return tokens_dec

    def forward(self, x_num):
        mu, logvar = self.encode(x_num)
        z = self.reparameterize(mu, logvar)
        tokens_rec = self.decode(z)
        x_num_rec = self.detokenizer(tokens_rec)
        return x_num_rec, mu, logvar

    def loss_function(self, x_num, x_num_rec, mu, logvar):
        recon_num = F.mse_loss(x_num_rec, x_num, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_num + self.beta * kl
        return loss, {"recon_num": recon_num, "kl": kl}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

d = 32
tokenizer = FeatureTokenizer(n_features=x_num.shape[1], d=d).to(device)
detokenizer = FeatureDetokenizer(tokenizer).to(device)
model = TransformerBetaVAE(tokenizer, detokenizer, d_model=d, latent_dim=16, beta=4.0).to(device)

x_num = x_num.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

batch_size = 256
num_epochs = 5
for epoch in range(num_epochs):
    perm = torch.randperm(x_num.size(0), device=device)
    x_num_shuf = x_num[perm]

    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    steps = 0

    for i in range(0, x_num.size(0), batch_size):
        xb_num = x_num_shuf[i:i+batch_size]

        x_num_rec, mu, logvar = model(xb_num)
        loss, logs = model.loss_function(xb_num, x_num_rec, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_recon += logs["recon_num"].item()
        running_kl += logs["kl"].item()
        steps += 1

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"loss {running_loss/steps:.4f} | "
        f"recon {running_recon/steps:.4f} | "
        f"kl {running_kl/steps:.4f}"
    )