from tinygrad import Tensor
from tinygrad.nn import Embedding, Linear

from .attention_head import AttentionHead


class GPT2:
  def __init__(self, voc_size: int, max_seq_len: int, emb_size: int, head_size: int) -> None:
    self.tok_emb = Embedding(voc_size, emb_size)
    self.tok_pos_emb = Embedding(max_seq_len, emb_size)
    self.attn = AttentionHead(emb_size, head_size, max_seq_len)
    self.lin = Linear(head_size, voc_size)
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
    return self.lin(self.attn(self.tok_emb(x) + self.tok_pos_emb(Tensor.arange(seq_len))))
