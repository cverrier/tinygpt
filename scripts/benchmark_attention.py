"""Benchmark MultiAttentionLayer vs BatchedMultiAttentionLayer."""

import io
import sys
import time

from tinygrad import Context, Device, Tensor, TinyJit
from tinygrad.helpers import GlobalCounters

from tinygpt.batched_multi_attention_layer import BatchedMultiAttentionLayer
from tinygpt.multi_attention_layer import MultiAttentionLayer


def benchmark_layer(layer, x: Tensor, warmup_iters: int, timed_iters: int) -> tuple[float, float]:
  """Benchmark a layer and return wall-clock and kernel-only times.

  Args:
    layer: The attention layer to benchmark.
    x: Input tensor (already realized on device).
    warmup_iters: Number of warmup iterations (not timed).
    timed_iters: Number of timed iterations.

  Returns:
    Tuple of (wall_clock_ms, kernel_only_ms) average times.
  """
  device = Device[Device.DEFAULT]

  @TinyJit
  def forward(inp: Tensor) -> Tensor:
    return layer(inp).realize()

  # Warmup: JIT compilation + kernel caching
  for _ in range(warmup_iters):
    forward(x)
    device.synchronize()

  # Timed iterations - wall-clock time
  total_wall_ns = 0
  for _ in range(timed_iters):
    device.synchronize()  # Ensure clean state
    st = time.perf_counter_ns()
    forward(x)
    device.synchronize()  # Wait for GPU completion
    total_wall_ns += time.perf_counter_ns() - st

  wall_clock_ms = (total_wall_ns / timed_iters) * 1e-6

  # Timed iterations - kernel-only time using hardware timestamps
  # NOTE: DEBUG=2 seems to be required for hardware timestamp collection on Metal
  # Redirect stdout to suppress verbose kernel output
  GlobalCounters.reset()
  old_stdout = sys.stdout
  sys.stdout = io.StringIO()
  try:
    with Context(DEBUG=2):
      for _ in range(timed_iters):
        forward(x)
        device.synchronize()
  finally:
    sys.stdout = old_stdout

  kernel_only_ms = (GlobalCounters.time_sum_s / timed_iters) * 1000

  return wall_clock_ms, kernel_only_ms


def main():
  # Configuration
  n_heads = 8
  head_size = 64
  emb_size = 512
  max_seq_len = 256
  batch_size = 32
  seq_len = 128
  warmup_iters = 10
  timed_iters = 100

  print("Benchmark: MultiAttentionLayer vs BatchedMultiAttentionLayer")
  print("=" * 80)
  print(f"n_heads={n_heads}, head_size={head_size}, emb_size={emb_size}")
  print(f"max_seq_len={max_seq_len}, batch_size={batch_size}, seq_len={seq_len}")
  print(f"warmup_iters={warmup_iters}, timed_iters={timed_iters}")
  print("=" * 80)

  # Create input tensor and realize it before timing
  Tensor.manual_seed(42)
  x = Tensor.randn(batch_size, seq_len, emb_size).realize()
  Device[Device.DEFAULT].synchronize()  # Ensure input is ready on device

  # Create layers (disable dropout for deterministic timing)
  multi = MultiAttentionLayer(n_heads, head_size, emb_size, max_seq_len, dropout_rate=0.0)
  batched = BatchedMultiAttentionLayer(n_heads, head_size, emb_size, max_seq_len, dropout_rate=0.0)

  # Benchmark MultiAttentionLayer
  print("\nBenchmarking MultiAttentionLayer...")
  multi_wall, multi_kernel = benchmark_layer(multi, x, warmup_iters, timed_iters)
  print(f"  Wall-clock: {multi_wall:.3f} ms, Kernel-only: {multi_kernel:.3f} ms")

  # Benchmark BatchedMultiAttentionLayer
  print("\nBenchmarking BatchedMultiAttentionLayer...")
  batched_wall, batched_kernel = benchmark_layer(batched, x, warmup_iters, timed_iters)
  print(f"  Wall-clock: {batched_wall:.3f} ms, Kernel-only: {batched_kernel:.3f} ms")

  # Summary
  print("\n" + "=" * 80)
  print("Summary")
  print("=" * 80)
  print(f"{'Layer':<30} {'Wall-clock':>12} {'Kernel-only':>12}")
  print("-" * 56)
  print(f"{'MultiAttentionLayer':<30} {multi_wall:>9.3f} ms {multi_kernel:>9.3f} ms")
  print(f"{'BatchedMultiAttentionLayer':<30} {batched_wall:>9.3f} ms {batched_kernel:>9.3f} ms")
  print("-" * 56)

  # Speedup based on wall-clock time
  if multi_wall > batched_wall:
    speedup = multi_wall / batched_wall
    print(f"BatchedMultiAttentionLayer is {speedup:.2f}x faster (wall-clock)")
  else:
    slowdown = batched_wall / multi_wall
    print(f"BatchedMultiAttentionLayer is {slowdown:.2f}x slower (wall-clock)")

  # Speedup based on kernel-only time
  if multi_kernel > batched_kernel:
    speedup = multi_kernel / batched_kernel
    print(f"BatchedMultiAttentionLayer is {speedup:.2f}x faster (kernel-only)")
  else:
    slowdown = batched_kernel / multi_kernel
    print(f"BatchedMultiAttentionLayer is {slowdown:.2f}x slower (kernel-only)")


if __name__ == "__main__":
  main()
