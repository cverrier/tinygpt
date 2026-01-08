from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear


class BatchedMultiAttentionLayer:
  def __init__(self, n_heads: int, head_size: int, emb_size: int, max_seq_len: int, dropout_rate: float = 0.2) -> None:
    self.n_heads = n_heads
    self.head_size = head_size
    self.qkv_proj = Linear(emb_size, 3 * n_heads * head_size, bias=False)
    self.tri_mask = Tensor.ones(max_seq_len, max_seq_len, dtype=dtypes.bool).tril().logical_not().requires_grad_(False)
    self.max_seq_len = max_seq_len
    self.dropout_rate = dropout_rate

  def __call__(self, x: Tensor) -> Tensor:
    """Return concatenated attention head outputs.

    Args:
      x: Tensor of shape (bs, seq_len, emb_size).
    Returns:
      Tensor of shape (bs, seq_len, n_heads * head_size).
    """
    bs, seq_len, _ = x.shape
    assert seq_len <= self.max_seq_len, "Sequence length exceeds maximum"

    # Project to (bs, seq_len, 3 * n_heads * head_size) and split into q, k, v
    qkv = self.qkv_proj(x)
    q, k, v = qkv.chunk(3, dim=-1)

    # Reshape to (bs, n_heads, seq_len, head_size)
    q = q.reshape(bs, seq_len, self.n_heads, self.head_size).transpose(1, 2)
    k = k.reshape(bs, seq_len, self.n_heads, self.head_size).transpose(1, 2)
    v = v.reshape(bs, seq_len, self.n_heads, self.head_size).transpose(1, 2)

    # Compute attention: (bs, n_heads, seq_len, head_size) @ (bs, n_heads, head_size, seq_len) -> (bs, n_heads, seq_len, seq_len)
    attn = (q @ k.transpose(-2, -1)) * k.shape[-1] ** -0.5
    attn = attn.masked_fill(self.tri_mask[:seq_len, :seq_len], float("-inf"))
    attn = attn.softmax(axis=-1).dropout(self.dropout_rate)

    # Apply attention to values: (bs, n_heads, seq_len, seq_len) @ (bs, n_heads, seq_len, head_size) -> (bs, n_heads, seq_len, head_size)
    out = attn @ v

    # Reshape back to (bs, seq_len, n_heads * head_size)
    return out.transpose(1, 2).reshape(bs, seq_len, self.n_heads * self.head_size)
