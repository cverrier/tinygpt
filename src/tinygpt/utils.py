from tinygrad import Tensor


def get_batch(X: Tensor, batch_size: int, seq_len: int) -> tuple[Tensor, Tensor]:
  return X[idxs := Tensor.randint(batch_size, high=len(X) - seq_len).reshape(-1, 1) + Tensor.arange(seq_len)], X[idxs + 1]
