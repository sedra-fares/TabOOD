import torch
import numpy as np
import pandas as pd
from models.diffusion_model import DenoisingMLP, GaussianDiffusion
from data_generation.BetaVAE import BetaVAE

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- VAE & DATA DIMENSIONS (MUST MATCH TRAINING) ---
# Adjust these values based on your previous training settings:
NUM_FEATURES = 10       # 'M': Number of features/columns (e.g., 10)
EMBED_DIM = 32          # 'd': Embedding dimension per feature (e.g., 32)
# The total latent dimension for diffusion is M * d
LATENT_DIM = NUM_FEATURES * EMBED_DIM  # 10 * 32 = 320 

# --- OOD GENERATION PARAMETERS ---
ALPHA = 0.4             # Mixing factor (0.0 = Pure Attack, 1.0 = Pure Normal)
N_SAMPLES = 1000        # Number of samples to generate

# --- CHECKPOINT PATHS ---
VAE_CHECKPOINT = "checkpoints/best_vae.pt"
DIFF_NORMAL_CKPT = "checkpoints/diffusion_normal.pt"
DIFF_ATTACK_CKPT = "checkpoints/diffusion_attack.pt"

# --- VAE INIT PARAMS ---
# You must initialize the VAE with the EXACT same structure as training
VAE_PARAMS = {
    'num_numerical': NUM_FEATURES,  
    'cat_cardinalities': [], # Add your categorical cardinalities list if used
    'd': EMBED_DIM,
    'beta': 1.0
}

def generate_ood_samples():
    print(f"--- Starting Generation on {DEVICE} ---")
    print(f"Target Latent Dim: {LATENT_DIM} (Reshaping to {NUM_FEATURES}x{EMBED_DIM})")

    # ---------------------------------------------------------
    # 1. LOAD DIFFUSION MODELS
    # ---------------------------------------------------------
    print("Loading Diffusion Models...")
    
    # Load Normal Model
    model_norm = DenoisingMLP(latent_dim=LATENT_DIM).to(DEVICE)
    model_norm.load_state_dict(torch.load(DIFF_NORMAL_CKPT, map_location=DEVICE))
    diff_norm = GaussianDiffusion(model_norm, device=DEVICE)

    # Load Attack Model
    model_att = DenoisingMLP(latent_dim=LATENT_DIM).to(DEVICE)
    model_att.load_state_dict(torch.load(DIFF_ATTACK_CKPT, map_location=DEVICE))
    diff_att = GaussianDiffusion(model_att, device=DEVICE)

    # ---------------------------------------------------------
    # 2. LOAD VAE MODEL
    # ---------------------------------------------------------
    print("Loading VAE...")
    vae = BetaVAE(**VAE_PARAMS)
    checkpoint = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    
    # Handle checkpoint structure (dictionary vs state_dict)
    if isinstance(checkpoint, dict) and "vae_state" in checkpoint:
        vae.load_state_dict(checkpoint["vae_state"])
    else:
        vae.load_state_dict(checkpoint)
        
    vae.to(DEVICE)
    vae.eval()

    # ---------------------------------------------------------
    # 3. GENERATION PROCESS
    # ---------------------------------------------------------
    print(f"Generating {N_SAMPLES} samples with Alpha={ALPHA}...")
    
    with torch.no_grad():
        # A. Sample from Latent Spaces (Pure Noise -> Latent Data)
        # Shape: [N_SAMPLES, 320]
        z_normal = diff_norm.sample(n_samples=N_SAMPLES, latent_dim=LATENT_DIM)

        # B. Sample from Latent Spaces (Attack -> Latent Data
        z_attack = diff_att.sample(n_samples=N_SAMPLES, latent_dim=LATENT_DIM)

        # B. OOD Interpolation (Linear Mixing)
        # Formula: z_ood = alpha * Normal + (1 - alpha) * Attack
        z_ood_flat = ALPHA * z_normal + (1 - ALPHA) * z_attack

        # C. RESHAPING (CRITICAL STEP) 
        # The VAE Decoder expects 3D input: [Batch, Sequence_Length, Embedding]
        # We transform [1000, 320] -> [1000, 10, 32]
        z_ood_3d = z_ood_flat.view(N_SAMPLES, NUM_FEATURES, EMBED_DIM)

        # D. Decode to Real Data Space
        # Note: Ensure your VAE decoder returns the reconstructed x (x_hat)
        x_gen = vae.decoder(z_ood_3d)

        # Move to CPU for saving
        x_gen_np = x_gen.cpu().numpy()

    # ---------------------------------------------------------
    # 4. SAVE RESULTS
    # ---------------------------------------------------------
    print("Saving to CSV...")
    if x_gen_np.ndim == 3:
        N, M, D = x_gen_np.shape
        print(f"Flattening 3D output {x_gen_np.shape} -> 2D {(N, M*D)}")
        x_gen_np = x_gen_np.reshape(N, -1)
    
    # Création des colonnes (feature_0 à feature_319)
    cols = [f"feature_{i}" for i in range(x_gen_np.shape[1])]
    
    df_ood = pd.DataFrame(x_gen_np, columns=cols)
    
    # Add a label column (e.g., 2 for 'OOD')
    df_ood['label'] = 2 
    
    save_path = "results/generated_ood_samples.csv"
    df_ood.to_csv(save_path, index=False)
    
    print(f"Success! Data saved to: {save_path}")
    print(f"Final Data Shape: {df_ood.shape}")

if __name__ == "__main__":
    generate_ood_samples()