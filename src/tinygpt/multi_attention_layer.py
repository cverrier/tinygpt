from tinygrad import Tensor

from .attention_head import AttentionHead


class MultiAttentionLayer:
  def __init__(self, n_heads: int, head_size: int, emb_size: int, max_seq_len: int) -> None:
    self.heads = [AttentionHead(emb_size, head_size, max_seq_len) for _ in range(n_heads)]
    self.max_seq_len = max_seq_len

  def __call__(self, x: Tensor) -> Tensor:
    """Return concatenated attention head outputs.

    Args:
      x: Tensor of shape (bs, seq_len, emb_size).
    Returns:
      Tensor of shape (bs, seq_len, n_heads * head_size).
    """
    _, seq_len, _ = x.shape
    assert seq_len <= self.max_seq_len, "Sequence length exceeds maximum"
    return Tensor.cat(*[head(x) for head in self.heads], dim=-1)
