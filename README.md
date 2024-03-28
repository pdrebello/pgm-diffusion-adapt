# Diffusion Adapt
This repository contains the [PyTorch](https://pytorch.org) source code to
reproduce the experiments in our NeurIPS2020 paper [Domain Adaptation as a Problem of Inference on Graphical Models
](https://arxiv.org/abs/2002.03278).

# Main code
* code_digits/diffusion_conditional_masked.py: train diffusion model, masking out labels of target domain
* code_digits/train_target.py: train classifier on target dataset
* code_digits/generator_generator.ipynb, code_digits/diffusion_generator.ipynb: generate synthetic images with generator of GAN, Diffusion model 

