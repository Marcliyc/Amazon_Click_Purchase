# Amazon EV/CM Implementation

This repository implements an EV/CM (Moe–Fader style) model for Amazon browsing and purchase sessions.

## Run

```bash
python -m src.evcm_pipeline \
  --input data/amazon_sessions.csv \
  --output reports/evcm \
  --visit-unit daily \
  --freq W \
  --n-sims 300 \
  --ev-starts 5 \
  --cm-starts 5 \
  --seed 123
```

## Tests

```bash
pytest
```
