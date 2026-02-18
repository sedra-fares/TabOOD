"""Simple linear OOD generation pipeline using pretrained beta-VAE and diffusion models.

Assumptions
- Diffusion models are saved TorchScript modules with a .sample(num_samples) method
  that returns latent vectors z of shape (N, d). You provide two class-conditional
  models: one for class 0 and one for class 1.
- The beta-VAE is a TorchScript module with a method to decode latents back to
  data space. Accepted method names (first one found): decode_latent(z),
  decode(z), or forward(z).
- Latent dimensionality matches between diffusion outputs and the beta-VAE.

Usage example
  python GeneratedComarison/GenerationPipeline.py \
	--vae_ckpt runs/beta_vae.pt \
	--diff0_ckpt runs/diffusion_class0.pt \
	--diff1_ckpt runs/diffusion_class1.pt \
	--n_samples 1000 \
	--alpha 0.8 \
	--out_csv generated_ood.csv

This will sample z0 ~ q(z|x, y=0) and z1 ~ q(z|x, y=1), interpolate
z = alpha*z0 + (1-alpha)*z1, and decode to data space.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import pandas as pd
import torch


def load_module(path: str, device: torch.device) -> torch.nn.Module:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Checkpoint not found: {path}")
	return torch.jit.load(path, map_location=device)


def decode_beta_vae(vae: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
	"""Call the first available decode-like method on the beta-VAE."""
	for name in ("decode_latent", "decode", "forward"):
		if hasattr(vae, name):
			return getattr(vae, name)(z)
	raise AttributeError("Beta-VAE does not expose decode_latent/decode/forward")


def sample_diffusion(diffusion: torch.nn.Module, n: int) -> torch.Tensor:
	if not hasattr(diffusion, "sample"):
		raise AttributeError("Diffusion model is expected to expose .sample(num_samples)")
	return diffusion.sample(n)


def generate_samples(
	vae: torch.nn.Module,
	diff0: torch.nn.Module,
	diff1: torch.nn.Module,
	n_samples: int,
	alpha: float,
	device: torch.device,
) -> torch.Tensor:
	vae.eval(); diff0.eval(); diff1.eval()
	with torch.no_grad():
		z0 = sample_diffusion(diff0, n_samples).to(device)
		z1 = sample_diffusion(diff1, n_samples).to(device)
		z = alpha * z0 + (1.0 - alpha) * z1
		x_hat = decode_beta_vae(vae, z)
	return x_hat


def main():
	parser = argparse.ArgumentParser(description="Generate OOD samples with pretrained beta-VAE and diffusion models.")
	parser.add_argument("--vae_ckpt", required=True, help="Path to TorchScript beta-VAE checkpoint")
	parser.add_argument("--diff0_ckpt", required=True, help="Path to TorchScript diffusion checkpoint for class 0")
	parser.add_argument("--diff1_ckpt", required=True, help="Path to TorchScript diffusion checkpoint for class 1")
	parser.add_argument("--n_samples", type=int, default=1000, help="Number of OOD samples to generate")
	parser.add_argument("--alpha", type=float, default=0.8, help="Interpolation weight for class-0 latent (alpha*z0 + (1-alpha)*z1)")
	parser.add_argument("--out_csv", required=True, help="Where to save generated samples (CSV)")
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
	args = parser.parse_args()

	device = torch.device(args.device)
	vae = load_module(args.vae_ckpt, device).to(device)
	diff0 = load_module(args.diff0_ckpt, device).to(device)
	diff1 = load_module(args.diff1_ckpt, device).to(device)

	x_hat = generate_samples(vae, diff0, diff1, args.n_samples, args.alpha, device)

	# Expect x_hat is a tensor shaped (N, D); convert to DataFrame and save
	if not isinstance(x_hat, torch.Tensor):
		raise TypeError("beta-VAE decode did not return a torch.Tensor")
	if x_hat.dim() != 2:
		x_hat = x_hat.view(x_hat.size(0), -1)

	df = pd.DataFrame(x_hat.cpu().numpy())
	os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
	df.to_csv(args.out_csv, index=False)
	print(f"Saved {len(df)} samples to {args.out_csv}")


if __name__ == "__main__":
	main()


### To Run: ###
# 1. Generate synthetic data using your model and save as CSV (must match real schema, including target column name).
# python GeneratedComarison/GenerationPipeline.py \
#   --vae_ckpt runs/beta_vae.pt \
#   --diff0_ckpt runs/diffusion_class0.pt \
#   --diff1_ckpt runs/diffusion_class1.pt \
#   --n_samples 1000 \
#   --alpha 0.8 \
#   --out_csv generated_ood.csv