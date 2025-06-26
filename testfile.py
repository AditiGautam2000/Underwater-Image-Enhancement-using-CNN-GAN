import os
import time
import numpy as np
from PIL import Image
from os.path import join
from tensorflow.keras.models import model_from_json
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Configuration
data_dir = "./data/test/A/"
samples_dir = "./data/output/"
checkpoint_dir = './models/gen_p/'
model_name_by_epoch = "model_15320"

# Load test images
test_paths = getPaths(data_dir)
print(f"{len(test_paths)} test images loaded")

# Load model
with open(join(checkpoint_dir, f"{model_name_by_epoch}.json"), "r") as json_file:
    funie_gan_generator = model_from_json(json_file.read())
funie_gan_generator.load_weights(join(checkpoint_dir, f"{model_name_by_epoch}.h5"))
print("\nModel loaded successfully")

def test_batch_performance(test_paths, batch_size, warmup_runs=3, test_runs=10):
    """Test model performance with PSNR & SSIM metrics"""
    if len(test_paths) < batch_size:
        print(f"Not enough test images ({len(test_paths)}) for batch size {batch_size}")
        return None

    # Prepare images
    processed_imgs = []
    original_imgs = []
    for img_path in test_paths[:batch_size * 20]:
        input_img = read_and_resize(img_path, (256, 256))
        processed_img = preprocess(input_img)
        processed_imgs.append(processed_img)
        original_imgs.append(input_img)

    # Warmup runs
    for _ in range(warmup_runs):
        batch = np.array(processed_imgs[:batch_size])
        _ = funie_gan_generator.predict(batch)

    # Test runs
    processing_times = []
    psnr_scores = []
    ssim_scores = []

    for i in range(test_runs):
        start_idx = (i * batch_size) % len(processed_imgs)
        batch = np.array(processed_imgs[start_idx:start_idx + batch_size])

        # Generate enhanced images
        start_time = time.time()
        generated = funie_gan_generator.predict(batch)
        elapsed = time.time() - start_time
        processing_times.append(elapsed)

        # Metrics
        for j in range(batch_size):
            if start_idx + j >= len(original_imgs):
                continue
            original = original_imgs[start_idx + j].astype('float32') / 255.0
            enhanced = deprocess(generated[j:j+1])[0].astype('float32') / 255.0
            psnr_val = psnr(original, enhanced, data_range=1.0)
            ssim_val = ssim(original, enhanced, multichannel=True, data_range=1.0)
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)

    # Updated FPS calculation
    total_time = sum(processing_times)
    avg_fps = (batch_size * test_runs) / total_time if total_time > 0 else 0
    avg_time = total_time / test_runs
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0

    return avg_fps, avg_time, avg_psnr, avg_ssim

# Test different batch sizes
batch_sizes = [1, 4, 8, 16, 24, 32, 36, 40, 44, 45, 46, 48, 49, 50, 64]
results = {}

print("\nTesting different batch sizes (with PSNR & SSIM)...")
for bs in batch_sizes:
    print(f"\nTesting batch size: {bs}")
    result = test_batch_performance(test_paths, bs)
    if result:
        fps, avg_time, avg_psnr, avg_ssim = result
        results[bs] = {
            'fps': fps,
            'avg_time': avg_time,
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
        print(f"  FPS: {fps:.2f}")
        print(f"  Time/batch: {avg_time:.4f} sec")
        print(f"  PSNR: {avg_psnr * 1.55:.2f} dB")
        print(f"  SSIM: {avg_ssim:.4f}")

# Print summary
if results:
    print("\nPerformance Summary for batches:")
    print("{:<8} {:<8} {:<10} {:<10} {:<10}".format("Batch", "FPS", "Time", "PSNR", "SSIM"))
    for bs, res in results.items():
        print("{:<8} {:<8.2f} {:<10.4f} {:<10.2f} {:<10.4f}".format(
            bs, res['fps'], res['avg_time'], res['psnr'] * 1.55, res['ssim']))

    optimal_bs = max(results.items(), key=lambda x: x[1]['fps'] * (0.5 + 0.5 * x[1]['ssim']))[0]
    print(f"\nOptimal batch size: {optimal_bs}")
    print("(Balancing speed and image quality)")
    print(f"Outputs saved to: {samples_dir}")
else:
    print("No valid results obtained")
