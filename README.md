## 1) Quick start (recommended)

From the repository root (the folder that contains `docker-compose.yml`):

```bash
# 1) Build & start everything
docker compose up --build

# 2) Stop (when finished)
# Press Ctrl+C, then:
docker compose down
```

If you want the services to keep running in the background:

```bash
docker compose up -d --build
docker compose logs -f
```

---

## 2) How to verify it is running

### Check containers
```bash
docker compose ps
```

### Check logs (server first)
```bash
docker compose logs -f server
```

You should see rounds progressing (e.g., `round=1`, `round=2`, …) and messages being produced/consumed.

---

## 3) Running specific experimental scenarios

The experiments described in the report differ by:
- attack mode (e.g., `shift` or `vehicles_to_car`)
- whether mitigation is enabled (detection + honeypot)
- timing (attack start round, detection round)

### 3.1 Configure via environment variables (preferred)
Many setups expose these values as env vars in `docker-compose.yml` under each service.  
Open **`docker-compose.yml`** and look for variables like:

- `ATTACK_MODE` (e.g., `shift`, `vehicles_to_car`)
- `ENABLE_DETECTION` (`1/0`)
- `ENABLE_HONEYPOT` (`1/0`)
- `ATTACK_START_ROUND` (e.g., `10`, `50`)
- `DETECTION_ROUND` (e.g., `10`, `50`)

> If your compose file uses different names, keep the *idea* the same: you are controlling the scenario through config values.

After changing configuration, restart:
```bash
docker compose down
docker compose up --build
```

### 3.2 Configure by editing code (alternative)
If env vars are not wired, the scenario is usually controlled in:
- `main.py` (server/orchestrator configuration)
- `benign_agent.py` / `malicious_agent.py` (agent behaviour)
- a config block/constants file (if present)

After editing code:
```bash
docker compose up --build
```

---

## 4) Scenarios used in the report (reference)

Below is a **mapping** you can use to reproduce the key cases.  
(Adjust exact values to match the variable names in your project.)

**Delayed attack activation**
- Attack start at round **10**: `ATTACK_START_ROUND=10`
- Attack start at round **50**: `ATTACK_START_ROUND=50`

**Delayed detection**
- Detection at round **10**: `DETECTION_ROUND=10`
- Detection at round **50**: `DETECTION_ROUND=50`

## 5) Outputs: where to find results

Depending on your implementation, results are typically written as:
- CSV files (per round accuracy / metrics)
- logs printed by `server` and agents
- plots generated from CSVs

Look for:
- a `shared\logs` folder
- files such as `*.csv`

---

## 6) Troubleshooting

### Kafka not ready / connection errors
Bring everything down, then up again:
```bash
docker compose down
docker compose up --build
```

### Port conflicts
If Kafka/Zookeeper ports are exposed and already in use, edit the host ports in `docker-compose.yml`.

### Low performance / out of memory
Close other applications, increase Docker memory limit, or reduce the number of agents/rounds.

---

## 7) Suggested “repro” checklist (for the evaluator)

1. `docker compose up --build`
2. Confirm rounds progress in `docker compose logs -f server`
3. Run at least:
   - benign-only baseline
   - shift attack without mitigation
   - shift attack with detection + honeypot
   - vehicles-to-car with attacker vision
4. Export logs / metrics files and generate plots