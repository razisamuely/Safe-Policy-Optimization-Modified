# Experiment 1: MACPO 8m Collision Baseline

## Grid Matrix
| Map | Algorithm | Cost Limit | Seeds | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `8m` | `MACPO` | 0.1 | 1, 2 | Strict constraint baseline. |
| `8m` | `MACPO` | 0.5 | 1, 2 | Moderate constraint baseline. |

## Execution Command
```bash
python safepo/multi_agent/macpo.py \
    --env-name 8m \
    --cost-type collision \
    --cost-limit 0.1 \
    --seed 1 \
    --write_terminal
```
