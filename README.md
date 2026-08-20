# sts2-agent

English | [简体中文](README.zh.md)

`sts2-agent` is an Agent/Mod prototype for *Slay the Spire 2*. It combines an in-game C# bridge mod, Python-side policies and orchestrator code, and local debugging tools for future LLM-driven autoplay.

## What This Repo Includes

- `src/sts2_agent/`: Python bridge client, policies, orchestrator, traces
- `mod/Sts2Mod.StateBridge/`: in-game STS2 bridge mod
- `mod/Sts2Mod.StateBridge.Host/`: local host for fixture/runtime-host validation
- `tests/`: Python unit tests
- `tools/`: build, install, validate, and live-debug scripts
- `CONVERTER.py`: fixed-size observation encoder, action masking, PPO model, bridge payload mapping
- `Train.py`: restartable PPO trainer with one worker per bridge endpoint

## Requirements

- Python 3.11+
- PyTorch and Requests
- .NET SDK 9
- Godot 4.5.1
- Windows install of *Slay the Spire 2*

## Quick Start

Build the mod against a real STS2 install:

```bash
dotnet build mod/Sts2Mod.StateBridge.sln \
  -p:Sts2ManagedDir="F:\\SteamLibrary\\steamapps\\common\\Slay the Spire 2\\data_sts2_windows_x86_64" \
  -p:Sts2ModLoaderDir="F:\\SteamLibrary\\steamapps\\common\\Slay the Spire 2\\data_sts2_windows_x86_64"
```

Install and launch the bridge mod:

```bash
python tools/debug_sts2_mod.py install --game-dir "F:\\SteamLibrary\\steamapps\\common\\Slay the Spire 2"
python tools/debug_sts2_mod.py debug --game-dir "F:\\SteamLibrary\\steamapps\\common\\Slay the Spire 2"
```

Run Python tests:

```bash
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

## Neural-network trainer

`Train.py` now runs actual on-policy PPO updates instead of inference-only autoplay. It also:

- automatically starts a fresh run from the bridge menu after a terminal run;
- advances reward/map/event/shop windows with safe defaults until those phases have policy features;
- keeps a rollout per bridge worker and updates one shared policy under a lock;
- saves model and optimizer state atomically so training can resume after interruption;
- tolerates stale decisions and temporary bridge outages without crashing the process.

### One game instance

Run the bridge with writes enabled, then start training:

```bash
python Train.py --base-url http://127.0.0.1:17654 --checkpoint checkpoints/ppo_spire_model.pt
```

Resume a previous run:

```bash
python Train.py \
  --base-url http://127.0.0.1:17654 \
  --resume checkpoints/ppo_spire_model.pt \
  --checkpoint checkpoints/ppo_spire_model.pt
```

### Multiple simultaneous instances

Parallel collection requires **one running game/bridge instance per endpoint**. The Python trainer does not clone Steam processes or share one bridge port between workers. Start separate instances with unique bridge ports, then pass each URL:

```bash
python Train.py \
  --base-url http://127.0.0.1:17654 \
  --base-url http://127.0.0.1:17655 \
  --base-url http://127.0.0.1:17656 \
  --checkpoint checkpoints/ppo_spire_model.pt
```

If your instances use sequential localhost ports, the shorthand is:

```bash
python Train.py --workers 3 --port-start 17654
```

Do not point multiple workers at the same port: they would control the same game and corrupt the decision/rollout sequence.

Useful limits for smoke tests:

```bash
python Train.py \
  --base-url http://127.0.0.1:17654 \
  --max-episodes 10 \
  --rollout-steps 64 \
  --checkpoint checkpoints/smoke.pt
```

The fixed observation remains 69 features: player vitals, ten hand slots, and five enemy slots. The policy has eleven actions: ten hand slots plus end turn. Card actions are resolved by stable card identifiers as well as hand indices, and target constraints are matched against alive enemy identifiers.

## Bridge API

Once loaded in-game, the bridge exposes:

- `GET /health`
- `GET /snapshot`
- `GET /actions`
- `POST /apply`
- `GET/POST/DELETE /agent-status`

Writes are disabled by default. Enable them explicitly before live action testing.

## Key Docs

- `docs/sts2-mod-local-development.md`: build, install, live debugging, and validation
- `docs/sts2-mod-upgrade-notes.md`: mod migration notes after game updates
- `docs/sts2-mod-agent-compatibility.md`: current bridge/runtime compatibility notes
- `docs/local-development.md`: local Python-side workflow notes
- `docs/prototype-validation.md`: fixture/prototype validation details

## Encoding Note

- Chinese docs and OpenSpec artifacts use UTF-8 without BOM.
- Avoid writing Chinese files through PowerShell text pipes; that may produce `???`.
