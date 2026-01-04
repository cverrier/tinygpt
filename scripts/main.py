from tinygrad import Tensor, TinyJit, dtypes, nn

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
model = GPT2(voc_size=voc_size, max_seq_len=max_seq_len, emb_size=32, head_size=16)
opt = nn.optim.AdamW(nn.state.get_parameters(model))


bs, seq_len = 4, 6


@TinyJit
@Tensor.train()
def train_step() -> Tensor:
  X, Y = get_batch(data_train, bs, seq_len)
  opt.zero_grad()
  loss = model(X).sparse_categorical_crossentropy(Y).backward()
  return loss.realize(*opt.schedule_step())


for i in range(10000):
  # GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
  loss = train_step()
  if i % 100 == 0:
    print(f"Step {i:3d} loss: {loss.item():6.2f}")
  # if i % 10 == 0:
  #   test_acc = get_test_acc().item()
  # t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")


# x, y = get_batch(data_train, bs, seq_len)
#
# logits = model(x)
# print("voc_size:", voc_size)
# print("logits shape:", logits.shape)
