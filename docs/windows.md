# brainbox on Windows

Target machine per spec: 8 GB RAM, dual/quad-core CPU, no GPU. brainbox runs
the from-scratch neurobrain checkpoints locally and is hard-capped at 3.8 GB
RSS (it refuses work above that; measured peak on Linux is ~300 MB with the
current 4.25M LM + 0.44M GripNet resident).

> Status 2026-10-22: everything below is exercised on Linux by tests except
> the Task Scheduler commands, which are standard `schtasks` invocations
> documented here and generated for you in M6 (`brainbox autostart`). If a
> step fails on real Windows, file it against agent 01 in the fleet repo
> (`agents/01-runtime/NOTES.md`) with the failing output.

## 1. Install Python

Python 3.11+ from https://www.python.org/downloads/windows/ — during setup
tick **“Add python.exe to PATH”**. Verify in a new PowerShell:

```powershell
py --version
```

## 2. Install brainbox

```powershell
py -m venv $env:USERPROFILE\brainbox-venv
& $env:USERPROFILE\brainbox-venv\Scripts\Activate.ps1

# CPU-only torch (the default index serves a ~550 MB CUDA build you don't need)
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch

# brainbox itself (77 KB wheel; from a git clone of neurobrain)
git clone https://github.com/dreamer2664/neurobrain.git $env:USERPROFILE\neurobrain
cd $env:USERPROFILE\neurobrain
pip install .
```

Alternative, no install: download `brainbox.pyz` (built by
`scripts/build_zipapp.py`) and run it with `py brainbox.pyz …` — it still
needs torch in whichever Python runs it.

## 3. Fetch the checkpoints

Checkpoints are release assets, never git commits:

```powershell
mkdir checkpoints -Force
$rel = "https://github.com/dreamer2664/neurobrain/releases/download/ckpt-1022"
Invoke-WebRequest "$rel/lm_best.pt"   -OutFile checkpoints\lm_best.pt
Invoke-WebRequest "$rel/grip_best.pt" -OutFile checkpoints\grip_best.pt

# Convert to checkpoint directories (sha256-verified layout brainbox loads)
python scripts\migrate_checkpoint.py checkpoints\lm_best.pt   --kind lm
python scripts\migrate_checkpoint.py checkpoints\grip_best.pt --kind grip
```

Expected sha256 (`Get-FileHash`): lm `0df9f2f6…`, grip `b449721d…`
(full values in `https://github.com/dreamer2664/neurobrain/releases/tag/ckpt-1022`).

## 4. First run

```powershell
brainbox doctor        # stack, tokenizer, both checkpoints -> verified=True
brainbox bench         # boot time, 20-token generation, peak RSS (2 threads)
brainbox serve         # local API on http://127.0.0.1:8321 (loopback only)
brainbox run "delete the tmp files" --script mytask.json
brainbox tasks         # task list
brainbox logs t-0001   # full history of one task
brainbox undo t-0001   # restore anything the task deleted
```

Data lives under `%USERPROFILE%\.brainbox\` (`tasks\`, `runs\`, `trash\`,
`inbox\`). Deletes performed by tasks go to `trash\<task-id>\` first and are
restorable byte-identically with `brainbox undo <task-id>`.

## 5. Autostart (Task Scheduler)

Run brainbox at logon so the watch loop survives reboots with its queue
intact:

```powershell
$action  = "`"$env:USERPROFILE\brainbox-venv\Scripts\brainbox.exe`" run --watch $env:USERPROFILE\.brainbox\inbox"
schtasks /Create /TN "brainbox" /TR $action /SC ONLOGON /RL LIMITED /F
schtasks /Query  /TN "brainbox"          # verify it registered
schtasks /Delete /TN "brainbox" /F       # remove it again
```

`run --watch` (M6) polls the inbox directory for task JSON files, executes
them through the drive loop, and parks unfinished work back on disk — killing
the process or rebooting loses nothing except the step in flight.

On Linux the equivalent is a `systemd --user` unit; M6 ships
`brainbox autostart install|uninstall|status` generating both.

## 6. Troubleshooting

- **`brainbox` not recognized** — activate the venv first, or call
  `& $env:USERPROFILE\brainbox-venv\Scripts\brainbox.exe …` directly.
- **doctor says `checkpoint NONE FOUND`** — run step 3 from the repo root,
  or set `$env:BRAINBOX_CHECKPOINT` to the migrated `checkpoints\lm` folder.
  Without a checkpoint brainbox still runs on a clearly-flagged stub
  (`verified` line in doctor output tells you which mode you are in).
- **port 8321 busy** — `$env:BRAINBOX_PORT = 8322` before `brainbox serve`.
- **antivirus flags `brainbox.pyz`** — use the pip install instead;
  .pyz is a plain zip of the repo's Python sources (`py -m zipfile -l`).
