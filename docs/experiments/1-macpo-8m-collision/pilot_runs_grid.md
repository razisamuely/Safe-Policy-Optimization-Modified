# Experiment 1: MACPO 8m Collision Baseline

## Grid Matrix
| Map | Algorithm | Cost Limit | Seeds | Job ID | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Map | Algorithm | Cost Limit | Seeds | Job ID | Status | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `8m` | `MACPO` | 0.1 | 1 | 16398801 | **RUNNING** | Strict constraint baseline. |
| `8m` | `MACPO` | 0.1 | 2 | 16398803 | **RUNNING** | Strict constraint baseline. |
| `8m` | `MACPO` | 0.5 | 1 | 16398805 | **RUNNING** | Moderate constraint baseline. |
| `8m` | `MACPO` | 0.5 | 2 | 16398812 | **RUNNING** | Moderate constraint baseline. |

## Execution Command
```bash
python safepo/multi_agent/macpo.py \
    --env-name 8m \
    --cost-type collision \
    --cost-limit 0.1 \
    --seed 1 \
    --write_terminal
```
