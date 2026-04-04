# word2kernel

Trimmed working repo for the DSA CuTeDSL kernel loop. This repo keeps the
agent loop, benchmark launchers, and reference corpus needed for `dsa_attention`
without dragging over the full FlashMLA tree.

## Bootstrap

Create or activate the local virtualenv, then install the loop/runtime deps:

```bash
.venv/bin/python -m pip install -r requirements-agent-loop.txt
```

The runtime uses the installed `openai-agents` package from `.venv`.
`openai-agents-python/` is kept in-repo as a reference copy for SDK exploration,
not as an import path hack.

## Environment

Create `.env` from `.env.example` and populate a real key:

```bash
cp .env.example .env
```

Required:

```bash
OPENAI_API_KEY=your_key_here
```

## Common Commands

Run the agent loop:

```bash
.venv/bin/python main.py --num-rounds 1
```

Run the fast synthetic checker:

```bash
.venv/bin/modal deploy scripts/bench_synthetic_service.py
.venv/bin/python scripts/bench_synthetic.py --solution-dir solution/dsa_attention
```

Run the full competition benchmark:

```bash
.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --lang python
```

## Repo Notes

- `dsa_attention` is the primary supported track in this trimmed repo.
- `solution/dsa_indexer/` is kept as a placeholder scaffold only.
- `tools/harness/` is self-contained again, but still expects a CUDA + PyTorch
  environment when you actually run the harness.
