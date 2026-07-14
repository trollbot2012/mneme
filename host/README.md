# Host adapters

Mneme's engine (`../mneme.py`) is host-agnostic. Host-specific policy and
plugin wiring live here and are **the source of truth** for production
deployments.

## Hermes (`host/hermes/`)

| File | Role |
|---|---|
| `provider.py` | Hermes `MemoryProvider` (deployed as `plugins/mneme/__init__.py`) |
| `runtime.py` | Session banks, write scope, recall evidence, episode finalize |
| `plugin.yaml` | Plugin metadata |
| `test_*.py` | Host-only lifecycle / policy tests |

### Deploy / verify

```powershell
# Deploy engine + adapter into a Hermes profile
python scripts/sync_hermes_bundle.py --profile $env:LOCALAPPDATA/hermes/profiles/aletheon

# Byte-identity check (engine + provider + runtime + yaml)
python scripts/sync_hermes_bundle.py --profile $env:LOCALAPPDATA/hermes/profiles/aletheon --check
python scripts/verify_hermes_bundle.py --profile $env:LOCALAPPDATA/hermes/profiles/aletheon
```

Do **not** edit the active profile plugin and leave the repo behind. Edit here,
sync, verify.

### Layout after sync

```
$HERMES_HOME/
  plugins/mneme/
    __init__.py   <- host/hermes/provider.py
    runtime.py    <- host/hermes/runtime.py
    mneme.py      <- repo-root mneme.py
    plugin.yaml
  mneme/          <- live store (db, notes, models) — not in this repo
```
