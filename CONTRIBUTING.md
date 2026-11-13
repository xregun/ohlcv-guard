# Contributing

Thanks for helping.

## Before you start

1. Open an issue for anything beyond a small fix, so the design can be agreed first.
2. Keep the policy: bad data raises and is never repaired; missing bars are reported, never invented. A new check is
   either an error (the data can't be trusted) or a warning (it can, but deserves a look).

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e .[dev]        # Windows: .venv\Scripts\pip
pytest --cov                           # coverage must stay at or above 95%
ruff check src tests
mypy
```

Tests use generated data only. Please don't add market data from any provider to the repository.

## Sign-off (DCO)

Every commit must be signed off: `git commit -s`. By signing off you certify the
[Developer Certificate of Origin 1.1](https://developercertificate.org/): that you wrote the change or have the right to
submit it under the project's license (Apache-2.0).
