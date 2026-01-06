# tinygpt

A minimal implementation of GPT architecture with [tinygrad](https://github.com/tinygrad/tinygrad), trained on [The Silmarillion](https://grokipedia.com/page/The_Silmarillion), written by [J. R. R. Tolkien](https://grokipedia.com/page/J._R._R._Tolkien).

This project is just for experimenting with tinygrad, and is not polished at all (and, very likely, might not be polished in the future).

If you want to run a simple experiment, make sure you have [uv](https://github.com/astral-sh/uv) installed, then run:
```bash
uv sync && uv run scripts/main.py
```

References:
- [Let's build GPT: from scratch, in code, spelled out -- Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Lord of The Rings - Text Data](https://github.com/jblazzy/LOTR)
