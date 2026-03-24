# Experiment History & Baseline Log

This log tracks the motivation and outcomes of the baseline experiments (MACPO, MAPPO-Lag, etc.) in this repository to compare against Safe Dreamers.

## 📅 Experiment Registry

| Experiment ID | Title | Status | Outcome | Key Lesson |
| :--- | :--- | :--- | :--- | :--- |
| **05** | [Collision Zero-Limit](./5-collision-zero-limit/) | **PENDING** | - | New strictly safe baseline. |
| **04** | [Collision Fine-Tuning](./4-collision-fine-tuning/) | **PENDING** | - | Ported Baseline (0.1/0.5). |
| **01** | [Initial Setup](./1-macpo-8m-collision/) | Completed | Success | Environment logic ported. |

---

## 📈 Technical Knowledge Base
- **Cost Alignment**: Ensure `collision_threshold=1.0` in `smac_wrapper.py`.
- **Environment**: All comparisons use `8m` map with `difficulty="7"` and `continuing_episode=True`.
