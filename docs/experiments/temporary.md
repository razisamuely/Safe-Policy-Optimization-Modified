# ⏳ Temporary Experiment Status (Pending Submission)

**Last Updated**: 2026-03-24
**Reason for Pause**: Slurm/Cluster instability.

## 🛠️ Work Completed (Ready to Go)

### **1. MACPO Baseline (Safe-Policy-Optimization)**
- **Code**: `collision` cost logic is fully ported and verified.
- **Sbatch Template**: Updated `sbatch_scripts/template_macpo.sbatch` with **GPU Keep-Alive** and proper cleanup traps.
- **Local Generation**: `submit_baseline.py` completed local generation of 6 jobs (Limits 0.0, 0.1, 0.5; Seeds 1, 2).
- **Archive**: `sbatch_transfer_spo.tar.gz` created locally with all `.sbatch` files and the keep-alive script.

### **2. Safe Dreamer (private-mamba)**
- **Sbatch Template**: Updated `sbatch_scripts/template.sbatch` with **GPU Keep-Alive** logic.
- **Local Generation**: `submit_experiments.py` is ready for `limit=0.0` (Seeds 1, 2).
- **Archive**: `sbatch_transfer_sd.tar.gz` created locally.

## 🚀 Resumption Steps (When Slurm is Ready)

1. **Transfer Archives**:
   ```bash
   # From Safe-Policy-Optimization:
   scp sbatch_transfer_spo.tar.gz razshmue@slurm.bgu.ac.il:workspace/Safe-Policy-Optimization-Modified/
   
   # From private-mamba:
   scp sbatch_transfer_sd.tar.gz razshmue@slurm.bgu.ac.il:workspace/private-mamba/
   ```

2. **Unpack on Cluster**:
   ```bash
   ssh razshmue@slurm.bgu.ac.il "cd workspace/Safe-Policy-Optimization-Modified && tar -xzf sbatch_transfer_spo.tar.gz"
   ssh razshmue@slurm.bgu.ac.il "cd workspace/private-mamba && tar -xzf sbatch_transfer_sd.tar.gz"
   ```

3. **Launch All 8 Jobs**:
   ```bash
   # SPO (6 Jobs)
   ssh razshmue@slurm.bgu.ac.il "cd workspace/Safe-Policy-Optimization-Modified && for f in sbatch_scripts/generated/macpo_8m_*.sbatch; do sbatch \$f; done"
   
   # SD (2 Jobs)
   ssh razshmue@slurm.bgu.ac.il "cd workspace/private-mamba && for f in sbatch_scripts/generated/safedreamer_*.sbatch; do sbatch \$f; done"
   ```

4. **Verify**: Check WandB for new runs syncing to the `private-mamba` project.
