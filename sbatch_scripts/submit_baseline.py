import subprocess
import os

def submit_experiments():
    template_path = "sbatch_scripts/template_macpo.sbatch"
    with open(template_path, "r") as f:
        template = f.read()

    cost_limits = [0.0, 0.1, 0.5]
    seeds = [1, 2]

    for limit in cost_limits:
        for seed in seeds:
            run_id = f"macpo_8m_{limit}_s{seed}"
            sbatch_path = f"sbatch_scripts/generated/{run_id}.sbatch"
            sbatch_content = template.replace("{COST_LIMIT}", str(limit)).replace("{SEED}", str(seed))
            
            with open(sbatch_path, "w") as f:
                f.write(sbatch_content)
            
            print(f"Generated {run_id} locally.")
            
            # Step 1: SCP the sbatch file to the remote server
            # remote_base = "workspace/Safe-Policy-Optimization-Modified"
            # remote_path = f"{remote_base}/{sbatch_path}"
            # scp_cmd = ["scp", sbatch_path, f"razshmue@slurm.bgu.ac.il:{remote_path}"]
            # subprocess.run(scp_cmd, check=True)

            # Step 2: SSH to the remote and run sbatch
            # ssh_cmd = [
            #     "ssh",
            #     "razshmue@slurm.bgu.ac.il",
            #     f"cd {remote_base} && sbatch {sbatch_path}"
            # ]
            # result = subprocess.run(ssh_cmd, check=True, capture_output=True, text=True)
            
            # job_id = result.stdout.strip().split()[-1] if result.stdout else "unknown"
            # print(f"Job submitted! Slurm ID: {job_id}")

if __name__ == "__main__":
    submit_experiments()
