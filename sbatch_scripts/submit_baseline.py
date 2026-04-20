import argparse
import csv
import os
import subprocess
from datetime import datetime


def submit_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["8m"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--cost_limits", type=float, nargs="+", default=[0.0])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    template_path = "sbatch_scripts/template_macpo.sbatch"
    remote_base = "workspace/Safe-Policy-Optimization-Modified"
    log_dir = "sbatch_scripts/generated"
    history_file = os.path.join(log_dir, "experiments_history.csv")
    os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(history_file):
        with open(history_file, "w", newline="") as f:
            csv.writer(f).writerow(["Timestamp", "Task", "CostLimit", "Seed", "JobID"])

    with open(template_path) as f:
        template = f.read()

    timestamp = datetime.now().strftime("date%m-%d-hr%H-%M-%S")

    for task in args.tasks:
        for limit in args.cost_limits:
            for seed in args.seeds:
                run_id = f"macpo_{task}_{limit}_s{seed}_{timestamp}"
                sbatch_path = f"sbatch_scripts/generated/{run_id}.sbatch"
                content = template.replace("{TASK}", task).replace("{COST_LIMIT}", str(limit)).replace("{SEED}", str(seed))

                with open(sbatch_path, "w") as f:
                    f.write(content)

                if args.dry_run:
                    print(f"[DRY-RUN] Generated {sbatch_path}")
                    continue

                remote_path = f"{remote_base}/{sbatch_path}"
                subprocess.run(["scp", sbatch_path, f"razshmue@slurm.bgu.ac.il:{remote_path}"], check=True)

                result = subprocess.run(
                    ["ssh", "razshmue@slurm.bgu.ac.il", f"cd {remote_base} && sbatch {sbatch_path}"],
                    check=True, capture_output=True, text=True,
                )
                job_id = result.stdout.strip().split()[-1] if result.stdout else "unknown"
                print(f"Submitted {run_id} → Slurm ID: {job_id}")

                with open(history_file, "a", newline="") as f:
                    csv.writer(f).writerow([timestamp, task, limit, seed, job_id])


if __name__ == "__main__":
    submit_experiments()
