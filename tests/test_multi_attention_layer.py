from tinygrad import Tensor

from tinygpt.batched_multi_attention_layer import BatchedMultiAttentionLayer
from tinygpt.multi_attention_layer import MultiAttentionLayer


def _copy_weights_to_batched(multi: MultiAttentionLayer, batched: BatchedMultiAttentionLayer) -> None:
  """Copy weights from MultiAttentionLayer to BatchedMultiAttentionLayer."""
  q_weights = Tensor.cat(*[head.query.weight for head in multi.heads], dim=0)
  k_weights = Tensor.cat(*[head.key.weight for head in multi.heads], dim=0)
  v_weights = Tensor.cat(*[head.value.weight for head in multi.heads], dim=0)
  batched.qkv_proj.weight = Tensor.cat(q_weights, k_weights, v_weights, dim=0)


class TestMultiAttentionLayerEquivalence:
  def test_same_output_for_single_sequence(self):
    Tensor.manual_seed(123)
    n_heads, head_size, emb_size, max_seq_len = 4, 16, 64, 32
    # Disable dropout to ensure deterministic outputs
    multi = MultiAttentionLayer(n_heads, head_size, emb_size, max_seq_len, dropout_rate=0.0)
    batched = BatchedMultiAttentionLayer(n_heads, head_size, emb_size, max_seq_len, dropout_rate=0.0)
    _copy_weights_to_batched(multi, batched)
    x = Tensor.randn(1, 10, emb_size)
    out_multi = multi(x)
    out_batched = batched(x)
    assert out_multi.allclose(out_batched)

  def test_same_output_for_batched_input(self):
    Tensor.manual_seed(456)
    n_heads, head_size, emb_size, max_seq_len = 4, 16, 64, 32
    multi = MultiAttentionLayer(n_heads, head_size, emb_size, max_seq_len, dropout_rate=0.0)
    batched = BatchedMultiAttentionLayer(n_heads, head_size, emb_size, max_seq_len, dropout_rate=0.0)
    _copy_weights_to_batched(multi, batched)
    x = Tensor.randn(8, 20, emb_size)
    out_multi = multi(x)
    out_batched = batched(x)
    assert out_multi.allclose(out_batched)

  def test_same_output_at_max_sequence_length(self):
    Tensor.manual_seed(789)
    n_heads, head_size, emb_size, max_seq_len = 2, 8, 32, 16
    multi = MultiAttentionLayer(n_heads, head_size, emb_size, max_seq_len, dropout_rate=0.0)
    batched = BatchedMultiAttentionLayer(n_heads, head_size, emb_size, max_seq_len, dropout_rate=0.0)
    _copy_weights_to_batched(multi, batched)
    x = Tensor.randn(4, max_seq_len, emb_size)
    out_multi = multi(x)
    out_batched = batched(x)
    assert out_multi.allclose(out_batched)
