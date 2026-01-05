from collections.abc import Callable

from tinygrad import Tensor
from tinygrad.nn import Linear

from .multi_attention_layer import MultiAttentionLayer


class TransformerBlock:
  def __init__(self, n_heads: int, head_size: int, emb_size: int, max_seq_len: int):
    self.layers: list[Callable[[Tensor], Tensor]] = [
      MultiAttentionLayer(n_heads, head_size, emb_size, max_seq_len),
      Linear(attn_out := n_heads * head_size, attn_out),
      Tensor.relu,
    ]

  def __call__(self, x: Tensor) -> Tensor:
    """Return the transformer block output.

    Args:
      x: Tensor of shape (bs, seq_len, emb_size).
    Returns:
      Tensor of shape (bs, seq_len, n_heads * head_size).
    """
    return x.sequential(self.layers)
