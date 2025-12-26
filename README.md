# VGGNet-Pytorch

[Korean Version](./README_KR.md)

This repository provides a clean PyTorch implementation of VGGNet (A, B, C, D, E) as described in the 2014 paper.

🚀 Results (CIFAR-10)
<img width="100%" alt="Final Result" src="https://github.com/user-attachments/assets/effba762-c043-4fa9-aa1c-0f5be827acf9" />

Key Achievement: Achieved stable convergence by applying Batch Normalization and a lower Learning Rate (0.0001).

## 🛠️ Quick Start
1. Requirements
Bash
```
conda env create -f environment.yml # Recommended
```
### or
```
pip install -r requirements.txt
```
## 2. Training
Bash
```
python train.py
```
## 🏗️ Architecture Summary
Mapped configurations from the original paper:

Python

vggnet_a_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
# ... Supports all configs (A-E)
<img width="618" alt="Architecture" src="https://github.com/user-attachments/assets/b32c8cf0-68fe-484b-a0c8-9d1c770c1659" /> <p align="center"><em>Table 1: VGGNet Configurations (Simonyan & Zisserman, 2014)</em></p>

## 📝 Conclusion
By integrating Batch Normalization, we successfully trained deep architectures (VGG-D, E) that were previously unlearnable due to vanishing gradients.
