# FUnIE-GAN: Fast Underwater Image Enhancement with Generative Adversarial Networks

**One-line:** Implementation of [FUnIE-GAN](https://arxiv.org/pdf/1903.09766.pdf) and related GAN-based architectures for underwater image enhancement, including training, evaluation, and benchmarking with PSNR/SSIM.

---

## Project Overview
Underwater images often suffer from color distortion, low contrast, and reduced visibility. This project implements **FUnIE-GAN** and supporting GAN architectures to enhance underwater images. It provides:

- **Model Architectures**  
  - `funieGAN.py`: Implementation of FUnIE-GAN with U-Net style generator, PatchGAN discriminator, and perceptual/content loss.  
  - `pix2pix.py`: Pix2Pix-based generator/discriminator (from UGAN).  
  - `resnet.py`: ResNet-based generator/discriminator (from UGAN).  

- **Training**  
  - `train_funieGAN.py`: End-to-end training pipeline with validation, checkpoint saving, and sample generation.  

- **Testing & Evaluation**  
  - `test_funieGAN2.py`: Loads trained models, benchmarks across different batch sizes, and reports **FPS, PSNR, SSIM**.  

- **Utilities**  
  - `data_utils.py`: Data loading, preprocessing, deprocessing utilities.  
  - `plot_utils.py`: Plotting/visualization utilities for generated samples and loss curves.  

---

## Tools & Technologies
- **Python**, **Google Colab**  
- **TensorFlow**, **Keras**  
- **OpenCV**, **NumPy**, **scikit-image**  
- **Matplotlib** for visualization  
- **VGG19** for perceptual content loss  

---

##  Key Features
- **U-Net + PatchGAN** architecture for fast underwater image enhancement.  
- **Lightweight FUnIE-GAN Generator** for real-time inference.  
- **PatchGAN Discriminator** for enforcing texture realism.  
- **Perceptual Loss with VGG19** for content fidelity.  
- **Validation Sampling** during training (Input ↔ Enhanced ↔ Ground-truth).  
- **Evaluation Metrics**: PSNR, SSIM, FPS for image quality & speed.  

---

##  Quantitative Evaluation
- **PSNR**: **42 dB**  
- **SSIM**: **0.815**  
- **FPS**: **14.3** (optimized batch inference)  

These results show significant improvements in **color, contrast, and clarity** for underwater images.  

---

##  Quick Start
### 1. Clone repo & install dependencies
```bash
git clone <repo-url>
cd funieGAN
pip install -r requirements.txt
```

### 2. Dataset preparation
Organize datasets as:
```
data/
  Paired/
    underwater_imagenet/
      trainA/   # distorted images
      trainB/   # ground-truth images
    underwater_dark/
  test/
    A/          # distorted test images
    B/          # ground-truth test images (if available)
```

### 3. Train model
```bash
python train_funieGAN.py
```
- Models & checkpoints saved in `checkpoints/funieGAN/<dataset>/`.  
- Validation samples saved in `data/samples/funieGAN/<dataset>/`.  

### 4. Test model
```bash
python test_funieGAN2.py
```
- Loads generator from checkpoints.  
- Runs performance benchmarking across batch sizes.  
- Outputs enhanced images + metrics to `data/output/`.  

---

## Example Outputs
- **Training samples**: Input ↔ Enhanced ↔ Ground-truth triplets.  
- **Testing**: Input ↔ Enhanced comparison.  
- **Metrics**: Average PSNR, SSIM, FPS across test runs.

---

## 📜 Reference
- Islam, Md Jahidul, et al. *Fast Underwater Image Enhancement for Improved Visual Perception*. arXiv:1903.09766 (2019).  
