from tinygrad import Tensor

from tinygpt.utils import get_batch


def test_get_batch():
  X = Tensor.arange(100)
  x, y = get_batch(X, batch_size=4, seq_len=8)
  assert x.shape == (4, 8)
  assert y.shape == (4, 8)
  assert (x[:, 1:] == y[:, :-1]).all().item()
