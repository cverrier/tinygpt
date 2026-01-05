from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear


class AttentionHead:
  def __init__(self, emb_size: int, head_size: int, max_seq_len: int) -> None:
    self.key = Linear(emb_size, head_size, bias=False)
    self.query = Linear(emb_size, head_size, bias=False)
    self.value = Linear(emb_size, head_size, bias=False)
    self.tri_mask = Tensor.ones(max_seq_len, max_seq_len, dtype=dtypes.bool).tril().logical_not().requires_grad_(False)

  def __call__(self, x: Tensor) -> Tensor:
    """Return the attention head output.

    Args:
      x: Tensor of shape (bs, seq_len, emb_size).
    Returns:
      Tensor of shape (bs, seq_len, head_size).
    """
    _, seq_len, emb_size = x.shape
    assert seq_len <= self.tri_mask.shape[0], "Sequence length exceeds maximum"
    k, q = self.key(x), self.query(x)
    return (q @ k.transpose(-2, -1) * emb_size**-0.5).masked_fill(self.tri_mask[:seq_len, :seq_len], float("-inf")).softmax(axis=-1) @ self.value(x)
