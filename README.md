# 🚀 TabOOD: Tabular Out-of-Distribution Data Synthesis

This repository contains the implementation and reproduction of the **TabOOD** framework. The project focuses on using generative AI (VAEs and Diffusion) to create synthetic tabular data that helps machine learning models handle "Out-of-Distribution" (OOD) scenarios—situations where the test data looks different from the training data.

---

## 🧠 Project Overview
Standard models fail when they encounter new patterns (like a new type of cyber-attack or a shift in demographic data). **TabOOD** solves this by:
1. Learning a compressed "latent" version of tabular data using a **Transformer-based VAE**.
2. Using **Score-based Diffusion Models** to learn the structure of that latent space.
3. Generating "boundary" samples (OOD) by interpolating between known classes.

   ## 📂 Repository Structure

* **`models/`**: Architecture for the Transformer VAE and the Denoising MLP.
* **`data_generation/`**: Logic for creating synthetic samples and OOD interpolation.
* **`loaders/`**: Data loading scripts for **NSL-KDD** and **ACS Income**.
* **`preprocessing/`**: Tools for data cleaning and feature tokenization.
* **`results/`**: Evaluation metrics and F1-score logs.
* **`train_VAE.ipynb`**: Notebook to train the latent representation.
* **`train_diffusion.ipynb`**: Notebook to train the diffusion generative model.

---

## 📊 Results & Performance
We tested the framework on two major datasets. Below are the baseline F1-scores on the OOD (Out-of-Distribution) test sets:

| Model | ACS Income (F1) | NSL-KDD (F1) |
| :--- | :---: | :---: |
| **Logistic Regression** | 0.7136 | 0.7435 |
| **Decision Tree** | 0.7092 | 0.7606 |
| **XGBoost** | 0.7396 | 0.7858 |

---

## 🛠️ Key Technical Features
* **Transformer VAE**: Uses self-attention to understand relationships between different tabular columns.
* **Latent Interpolation**: A mathematical way to "mix" classes in the latent space to create difficult edge cases for the model to learn from.
* **$\beta$-VAE Regularization**: Forces the model to learn a structured, smooth latent space.

---

## 📝 Citation
This work is a reproduction based on:

> **Puranik, B., Can, B., & Fan, Y. (2025).** *Tabular out-of-distribution data synthesis for enhancing robustness.* Association for the Advancement of Artificial Intelligence (AAAI) / arXiv.

---

## 📄 Project Documentation
You can find the detailed analysis and reports here:

* **[Our Reproduction Paper](./Revisited%20Tabular%20out-of-distribution%20data%20synthesis%20for%20enhancing%20robustness.pdf)** - Detailed report of our work and findings.
* **[Original TabOOD Paper](./Tabular%20out-of-distribution%20data%20synthesis%20for%20enhancing%20robustness.pdf)** - The primary study we based our reproduction on.

---
