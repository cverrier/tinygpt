from tinygrad import Tensor, dtypes


def generate(model, tokens: list[int], max_new_tokens: int) -> list[int]:
  with Tensor.train(False):
    for _ in range(max_new_tokens):
      ctx = tokens[-model.max_seq_len :]
      x = Tensor([ctx], dtype=dtypes.int)
      next_tok = model(x)[:, -1, :].softmax().multinomial().item()
      tokens.append(next_tok)
  return tokens
