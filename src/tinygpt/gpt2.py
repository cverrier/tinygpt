from collections.abc import Callable

from tinygrad import Tensor
from tinygrad.nn import Embedding, LayerNorm, Linear

from .transformer_block import TransformerBlock


class GPT2:
  def __init__(self, voc_size: int, max_seq_len: int, emb_size: int, n_blocks: int, n_heads: int, head_size: int, dropout_rate: float = 0.2) -> None:
    self.tok_emb = Embedding(voc_size, emb_size)
    self.tok_pos_emb = Embedding(max_seq_len, emb_size)
    self.layers: list[Callable[[Tensor], Tensor]] = (
      [
        TransformerBlock(n_heads, head_size, emb_size, max_seq_len, dropout_rate),
      ]
      + [TransformerBlock(n_heads, head_size, n_heads * head_size, max_seq_len, dropout_rate) for _ in range(n_blocks - 1)]
      + [
        LayerNorm(emb_size),
        Linear(emb_size, voc_size),
      ]
    )
    self.max_seq_len = max_seq_len

  def __call__(self, x: Tensor) -> Tensor:
    """Return logits.

    Args:
      x: Tensor of shape (bs, seq_len) with token indices.
    Returns:
      Tensor of shape (bs, seq_len, voc_size) with logits.
    """
    bs, seq_len = x.shape
    assert seq_len <= self.max_seq_len, "Sequence length exceeds maximum"
    return (self.tok_emb(x) + self.tok_pos_emb(Tensor.arange(seq_len))).sequential(self.layers)
