
import tensorflow as tf
import time

# Load your model (replace with your actual model)
model = tf.keras.applications.ResNet50(weights=None)  # Example model
batch_sizes = [1, 2, 4, 8, 16, 32,40,45,50,55, 64]
fps_results = {}

# Disable any unnecessary overhead
tf.config.optimizer.set_jit(True)  # Enable XLA compilation for better performance

for bs in batch_sizes:
    # Create dummy input (adjust shape to match your model's input)
    dummy_input = tf.random.normal((bs, 224, 224, 3))  # Note: TensorFlow uses channels-last
    
    # Warm-up runs
    for _ in range(10):
        _ = model(dummy_input)
    
    # Synchronize and time the inference
    start_time = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    
    # Ensure all GPU operations are finished
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.sync_devices()
    
    elapsed = time.time() - start_time
    fps = 100 * bs / elapsed  # Frames per second
    
    # Get GPU memory stats (if available)
    try:
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        used_mem = mem_info['current'] / (1024 ** 2)  # Convert to MB
    except:
        used_mem = 0
    
    fps_results[bs] = fps
    print(f"Batch Size: {bs}, FPS: {fps:.2f}, GPU Mem: {used_mem:.2f} MB")

# Find the batch size with maximum FPS
optimal_batch = max(fps_results, key=fps_results.get)
print("\nOptimal Batch Size:", optimal_batch)

