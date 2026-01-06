from tinygrad import Tensor, TinyJit, dtypes, nn

from tinygpt.generate import generate
from tinygpt.gpt2 import GPT2
from tinygpt.utils import get_batch

with open("data/1-silmarillion/full.txt", "r", encoding="utf-8") as f:
  text = f.read()

chars = sorted(set(text))
voc_size = len(chars)

token_to_idx, idx_to_token = {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}


def encode_text(text: str) -> list[int]:
  return [token_to_idx[c] for c in text]


def decode_text(encoded: list[int]) -> str:
  return "".join(idx_to_token[i] for i in encoded)


assert text == decode_text(encoded := encode_text(text))

# Train/Val split (90/10)
data_train = Tensor(encoded[:612857], dtype=dtypes.int)
data_val = Tensor(encoded[612857:], dtype=dtypes.int)


max_seq_len = 8
emb_size = 32
n_heads = 4
head_size = emb_size // n_heads
dropout_rate = 0.2
model = GPT2(voc_size=voc_size, max_seq_len=max_seq_len, emb_size=emb_size, n_heads=n_heads, head_size=head_size, dropout_rate=dropout_rate)
opt = nn.optim.AdamW(nn.state.get_parameters(model))


bs, seq_len = 32, 6


@TinyJit
@Tensor.train()
def train_step() -> Tensor:
  X, Y = get_batch(data_train, bs, seq_len)
  opt.zero_grad()
  loss = model(X).sparse_categorical_crossentropy(Y).backward()
  return loss.realize(*opt.schedule_step())


@TinyJit
@Tensor.train(False)
def estimate_train_loss(batch_size: int) -> Tensor:
  X, Y = get_batch(data_train, batch_size, seq_len)
  loss = model(X).sparse_categorical_crossentropy(Y)
  return loss.realize()


@TinyJit
@Tensor.train(False)
def estimate_val_loss(batch_size: int) -> Tensor:
  X, Y = get_batch(data_val, batch_size, seq_len)
  loss = model(X).sparse_categorical_crossentropy(Y)
  return loss.realize()


print("*****")
print("Token generation before training")
toks = [33]
generate(model, toks, max_new_tokens=100)
print(decode_text(toks))

n_iters = 10000
eval_bs = 8 * bs
for i in range(n_iters):
  loss = train_step()
  if i % 100 == 0:
    print(f"Step {i:3d} loss: {loss.item():6.2f}")
    estim_train_loss = estimate_train_loss(eval_bs)
    estim_val_loss = estimate_val_loss(eval_bs)
    print(f"  train loss: {estim_train_loss.item():6.2f}, val loss: {estim_val_loss.item():6.2f}")

print("*****")
print("Token generation after training")
toks = [33]
generate(model, toks, max_new_tokens=100)
print(decode_text(toks))
