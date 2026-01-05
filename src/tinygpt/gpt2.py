from collections.abc import Callable

from tinygrad import Tensor
from tinygrad.nn import Embedding, Linear

from .multi_attention_head import MultiAttentionHead


class GPT2:
  def __init__(self, voc_size: int, max_seq_len: int, emb_size: int, n_heads: int, head_size: int) -> None:
    self.tok_emb = Embedding(voc_size, emb_size)
    self.tok_pos_emb = Embedding(max_seq_len, emb_size)
    self.attn_heads = MultiAttentionHead(n_heads, emb_size, head_size, max_seq_len)
    self.ffwd: list[Callable[[Tensor], Tensor]] = [Linear(attn_out := n_heads * head_size, attn_out), Tensor.relu]
    self.lin = Linear(attn_out, voc_size)
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
    return self.lin(self.attn_heads(self.tok_emb(x) + self.tok_pos_emb(Tensor.arange(seq_len))).sequential(self.ffwd))
