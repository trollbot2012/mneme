# Repository layout

```
Mneme/
  mneme.py                 # Host-agnostic engine (single-file doctrine)
  host/
    README.md
    hermes/                # Source of truth for the Hermes plugin
      provider.py          # -> $HERMES_HOME/plugins/mneme/__init__.py
      runtime.py           # -> .../runtime.py
      plugin.yaml
      test_*.py
  scripts/
    sync_hermes_bundle.py  # Deploy engine + host adapter into a profile
    verify_hermes_bundle.py
    archive_thin_episodes.py
    provision_embeddings.{ps1,sh}
  docs/
    ADR-0004-...
    ADR-0005-...
    ADR-0006-episode-quality-and-host-layout.md
  test_*.py                # Engine-level suite (stdlib, no Hermes)
  CONTEXT.md               # Domain language
  PRD.md / README.md / HANDOFF.md
```

## Deploy rule

1. Edit `mneme.py` and/or `host/hermes/*` in this repo.
2. `python scripts/sync_hermes_bundle.py --profile <HERMES_HOME>`
3. `python scripts/verify_hermes_bundle.py --profile <HERMES_HOME>`

Never treat the active profile plugin as source of truth.
