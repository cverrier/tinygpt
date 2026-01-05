from collections.abc import Callable

from tinygrad import Tensor
from tinygrad.nn import LayerNorm, Linear

from .multi_attention_layer import MultiAttentionLayer


class TransformerBlock:
  def __init__(self, n_heads: int, head_size: int, emb_size: int, max_seq_len: int):
    self.norm_mal_proj: list[Callable[[Tensor], Tensor]] = [
      LayerNorm(emb_size),
      MultiAttentionLayer(n_heads, head_size, emb_size, max_seq_len),
      Linear(n_heads * head_size, emb_size),
    ]
    self.norm_ffwd: list[Callable[[Tensor], Tensor]] = [
      LayerNorm(emb_size),
      Linear(emb_size, 4 * emb_size),
      Tensor.relu,
      Linear(4 * emb_size, emb_size),
    ]

  def __call__(self, x: Tensor) -> Tensor:
    """Return the transformer block output.

    Args:
      x: Tensor of shape (bs, seq_len, emb_size).
    Returns:
      Tensor of shape (bs, seq_len, emb_size).
    """
    return (xx := (x + x.sequential(self.norm_mal_proj))) + xx.sequential(self.norm_ffwd)
