#!/usr/bin/env python3
"""
GPU Keep-Alive Script

Prevents SLURM GPU idle detection by periodically using GPU resources.
This script runs in the background during training to ensure the GPU
is utilized at least every 30 minutes, preventing the 4-hour idle threshold
from triggering job cancellation.

Target: ~50% GPU utilization for 10 seconds every 30 minutes

Usage:
  python scripts/gpu_keepalive.py &
  # Returns PID for tracking
"""

import torch
import time
import sys
import os
from datetime import datetime

def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[GPU-KeepAlive {timestamp}] {message}", flush=True)

def stress_gpu(duration_seconds=10, target_utilization=0.5):
    """
    Stress GPU for specified duration to achieve target utilization.

    Args:
        duration_seconds: How long to stress the GPU
        target_utilization: Target GPU utilization (0.0 to 1.0)
    """
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available!")
        return False

    device = torch.device("cuda:0")
    log(f"Starting GPU stress on {torch.cuda.get_device_name(0)}")

    try:
        # Create large matrices to stress GPU
        # Adjust size based on GPU memory (using ~2GB)
        matrix_size = 8192
        log(f"Creating {matrix_size}x{matrix_size} matrices...")

        # Allocate tensors
        A = torch.randn(matrix_size, matrix_size, device=device)
        B = torch.randn(matrix_size, matrix_size, device=device)

        start_time = time.time()
        iterations = 0

        # Perform matrix operations for specified duration
        while time.time() - start_time < duration_seconds:
            # Matrix multiplication is GPU-intensive
            C = torch.matmul(A, B)

            # Additional operations to maintain utilization
            C = C + A
            C = torch.relu(C)
            C = C * 0.999  # Prevent overflow

            # Update matrices to prevent optimization
            A = C

            iterations += 1

            # Control utilization by adding small sleep
            # Lower target_utilization = longer sleep
            sleep_time = (1.0 - target_utilization) * 0.01
            if sleep_time > 0:
                time.sleep(sleep_time)

        elapsed = time.time() - start_time
        log(f"Completed {iterations} iterations in {elapsed:.2f}s")

        # Clean up
        del A, B, C
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        log(f"ERROR during GPU stress: {e}")
        return False

def main():
    """Main loop: stress GPU every 30 minutes."""
    log("GPU Keep-Alive started")
    log(f"PID: {os.getpid()}")

    # Configuration
    INTERVAL_MINUTES = 60  # 1 hour interval
    STRESS_DURATION_SECONDS = 10
    TARGET_UTILIZATION = 0.5

    log(f"Configuration:")
    log(f"  Interval: {INTERVAL_MINUTES} minutes")
    log(f"  Stress duration: {STRESS_DURATION_SECONDS} seconds")
    log(f"  Target utilization: {TARGET_UTILIZATION * 100}%")

    # Check CUDA availability
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available. Exiting.")
        sys.exit(1)

    log(f"CUDA available: {torch.cuda.get_device_name(0)}")

    iteration = 0

    try:
        while True:
            iteration += 1
            log(f"=== Iteration {iteration} ===")

            # Stress GPU
            success = stress_gpu(
                duration_seconds=STRESS_DURATION_SECONDS,
                target_utilization=TARGET_UTILIZATION
            )

            if not success:
                log("WARNING: GPU stress failed, continuing anyway...")

            # Wait for next iteration
            log(f"Sleeping for {INTERVAL_MINUTES} minutes...")
            time.sleep(INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        log("Received interrupt signal, shutting down...")
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        sys.exit(1)

    log("GPU Keep-Alive stopped")

if __name__ == "__main__":
    main()
