# Flower Federated Setup (No-KD student baseline)

This repository now contains a minimal Flower integration to run **federated FedAvg** over the **Student** model using the existing **no-KD** (supervised) path.

Why start with no-KD?
- It avoids shipping a Teacher model to every client.
- It validates that dataset partitioning, parameter exchange, and aggregation work.

## Install

```bash
pip install -r requirements.txt
```

This adds `flwr` to dependencies.

## Run (multi-process)

### 1) Start the server

```bash
python -m federated.flower_server \
  --server-address 0.0.0.0:8080 \
  --num-rounds 5 \
  --min-available-clients 2 \
  --min-fit-clients 2 \
  --min-evaluate-clients 2 \
  --local-epochs 1 \
  --lr 3e-4
```

### 2) Start clients (in separate terminals)

Client 0:
```bash
python -m federated.flower_client \
  --config config/no-kd/federated-flower.yaml \
  --cid 0 --num-clients 2 \
  --server-address 127.0.0.1:8080 \
  --device cpu
```

Client 1:
```bash
python -m federated.flower_client \
  --config config/no-kd/federated-flower.yaml \
  --cid 1 --num-clients 2 \
  --server-address 127.0.0.1:8080 \
  --device cpu
```

## Notes

- Dataset partitioning is **deterministic** based on `seed` in the YAML. Each split (train/dev/test) is partitioned independently.
- Each client writes logs under:
  - `logs/no-kd/federated-flower/federated/client_{cid}/`
- This is a baseline scaffold. Extending it to **federated distillation** can be done next by:
  - deciding whether the Teacher is global (server-side) or local (client-side)
  - exchanging student weights only, or exchanging projected features/statistics
