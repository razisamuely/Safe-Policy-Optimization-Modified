# Experiment History & Baseline Log

This log tracks the motivation and outcomes of the baseline experiments (MACPO, MAPPO-Lag, etc.) in this repository to compare against Safe Dreamers.

## 📅 Experiment Registry

| Experiment ID | Title | Status | Outcome | Key Lesson |
| :--- | :--- | :--- | :--- | :--- |
| **01** | [MACPO 8m Collision](./1-macpo-8m-collision/) | **PLANNED** | TBD | Evaluation of reactive baseline on collision cost. |

---

## 📈 Technical Knowledge Base
- **Cost Alignment**: Ensure `collision_threshold=1.0` in `smac_wrapper.py`.
- **Environment**: All comparisons use `8m` map with `difficulty="7"` and `continuing_episode=True`.
