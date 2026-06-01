# Headless Cloud Execution (IB Gateway + IBC on a VM)

Trade execution runs on a small always-on cloud VM, fully decoupled from the local
signal-generation workstation. The workstation only produces the daily signal; the
VM pulls it at market open and executes it — no desktop trading terminal, no laptop
needing to stay awake, and no broker API socket silently disabling itself on the
terminal's daily restart.

```
Workstation (evening)                  GCS                     VM (always-on)
 daily pipeline -> signal CSV  --push-->  bucket  --pull(09:31 ET)-->  executor
                                                                        |  ib_insync
                                                                        v
                                                              ib-gateway (IBC) -> broker
```

## What's in this folder
| File | Runs where | Purpose |
|------|-----------|---------|
| `docker-compose.yml` | VM | Defines `ib-gateway` (IBC) + `executor` services |
| `Dockerfile` / `requirements.txt` | VM | Builds the lightweight executor (no ML stack) |
| `entrypoint.sh` / `crontab` | VM | Keeps executor alive; fires trade at market open Mon–Fri |
| `pull_signal_and_trade.sh` | VM | Downloads signal from object storage, runs trader, unified logging |
| `gcs_pull.py` | VM | Tiny object-storage download helper |
| `gcs_push_signal.py` | Workstation | Uploads the signal after the evening pipeline |
| `.env.template` | VM | Copy to `.env`, fill broker creds + bucket (never commit) |

> The trader reads `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID`, `SIGNAL_FILE`,
> `TRADE_LOG_FILE`, and `MAX_SIGNAL_AGE_BDAYS` from the environment, so the same
> code runs locally (defaults to `127.0.0.1`) and inside the container
> (`ib-gateway` on the internal Docker network).

---

## One-time setup

### A. Cloud project + bucket + service account
```bash
gcloud config set project YOUR_PROJECT
gsutil mb -l YOUR_REGION gs://YOUR_SIGNALS_BUCKET

# Least-privilege service account scoped to the single signal bucket
gcloud iam service-accounts create signal-exec
gsutil iam ch serviceAccount:signal-exec@YOUR_PROJECT.iam.gserviceaccount.com:objectAdmin \
    gs://YOUR_SIGNALS_BUCKET
gcloud iam service-accounts keys create gcs-sa.json \
    --iam-account signal-exec@YOUR_PROJECT.iam.gserviceaccount.com
```

### B. Create the VM (small + always-on)
```bash
gcloud compute instances create signal-exec-vm \
    --machine-type e2-small \
    --zone YOUR_ZONE \
    --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size 20GB
```
Then SSH in and install Docker + the Compose plugin (use Docker's official apt
repository so the `docker-compose-plugin` package is available):
```bash
# add Docker's official repo, then:
sudo apt-get update && sudo apt-get install -y docker-ce docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker
```

### C. Copy this folder to the VM and add secrets
```bash
cd deploy
cp .env.template .env
# edit .env: broker user/password, GCS_BUCKET, TRADING_MODE, IB_PORT, etc.
mkdir -p secrets data logs
# place the service-account key at secrets/gcs-sa.json (scp it in; never commit)
```
The trader is baked into the image at build time, so copy the current trader in
before building:
```bash
cp ../src/trade_execution/spy_timing_trader.py ./spy_timing_trader.py
```

### D. Build + launch
```bash
docker compose build
docker compose up -d
docker compose logs -f ib-gateway      # watch for successful login
```
First login may require broker 2FA via the mobile app — approve it once. For fully
unattended daily restarts, configure the broker's app-based ("IB Key") 2FA.

---

## Daily flow (after setup)
1. The evening pipeline writes the signal CSV, then uploads it:
   ```bash
   python deploy/gcs_push_signal.py      # GCS_BUCKET set in the environment
   ```
2. The VM's in-container cron pulls the signal at market open and trades
   automatically. The workstation can be offline.

## Verify it works (paper first)
Set `TRADING_MODE=paper` and the paper `IB_PORT` in `.env`, then:
```bash
docker compose exec executor /app/pull_signal_and_trade.sh   # manual dry-run
docker compose exec executor tail -n 50 /var/log/spy/executor.log
```
Once a paper trade completes cleanly end-to-end, switch to `live`.

## Why this design is robust
- **IBC re-enables the API socket on every daily Gateway restart**, so the API
  can't silently stay disabled the way a GUI desktop terminal can.
- The VM is always on — no sleep, no OS-update reboot, no lost overnight session.
- Generation and execution share nothing but a versioned signal object in cloud
  storage, so a failure on one side can't corrupt the other.
- A business-day staleness guard refuses signals older than `MAX_SIGNAL_AGE_BDAYS`
  (weekend-aware), so a missed run can't fire a stale trade.
- All outcomes land in `logs/executor.log` + `logs/errors.log`.

## Security notes
- The broker API port is **not** published to the host or internet — only the
  executor reaches it over the internal Docker network. Do not add a `ports:`
  mapping for it.
- `.env`, `secrets/`, and `*.json` keys are git-ignored. The real broker password
  lives only in the VM's `.env`. Rotate the password and service-account key if
  ever exposed.
- Start in **paper** mode and only promote to live after a clean fill.
