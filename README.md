# Denoising Diffusion Probabilistic Models (DDPM)

This repository contains an implementation of **Denoising Diffusion Probabilistic Models (DDPM)** — a powerful class of generative models that learn data distributions through a sequence of denoising steps.

## 🧠 Overview

DDPMs are a class of generative models inspired by nonequilibrium thermodynamics. They work by gradually adding Gaussian noise to data and then learning to reverse this process to generate new samples.

This implementation follows the approach described in the original paper:

> **Denoising Diffusion Probabilistic Models**  
> *Jonathan Ho, Ajay Jain, Pieter Abbeel*  
> [[Paper](https://arxiv.org/abs/2006.11239)]

---

## 📁 Project Structure

ddpm/
├── Dataset folder/                # Dataset directory with class-wise image folders
│   └── Class/
│       └── class_images/         # Images for a specific class
│
├── models/                       # Model definitions
│   └── DDPM_Unconditional/
│       └── unet.py               # UNet architecture for unconditional DDPM
│
├── generated/                    # Output directory for generated/sampled images
│   └── sampled images/           # Generated image results
│
├── Diffusion.py                  # Core diffusion process (forward & reverse)
├── sample.py                     # Script to sample images using a trained model
├── unet.py 