# optuna_safepo_macpo.py
# Usage: python optuna_safepo_macpo.py
import os, subprocess, time, re, csv, argparse
from datetime import datetime
import optuna

## === USER SETTINGS: edit these ===
SAFEPO_ROOT = "/path/to/Safe-Policy-Optimization"   # << set this
TRAIN_SCRIPT = os.path.join(SAFEPO_ROOT, "safepo", "multi_agent", "macpo.py")
EVAL_SCRIPT  = os.path.join(SAFEPO_ROOT, "safepo", "evaluate.py")
TASK = "Safety2x4AntVelocity-v0"    # change to your SMAC wrapper task if needed
TOTAL_STEPS_PER_TRIAL = 200000     # fast proxy budget per trial (tune upward later)
N_TRIALS = 200
N_JOBS = 2                         # parallel trials (careful with GPUs)
STUDY_NAME = "macpo_optuna_run"
COST_LIMIT = 10.0                  # change to your constraint threshold
BETA = 1.0                         # penalty weight for constraint violations
BASE_SEED = 1000
## =================================

def parse_metric_from_log(logpath):
    """Try several heuristics to extract (reward, cost) from the training log or runs files."""
    reward = None; cost = None
    txt = ""
    try:
        txt = open(logpath, "r", encoding="utf-8", errors="ignore").read()
    except:
        txt = ""
    # common heuristic patterns (adjust regexes if your logs differ)
    m = re.search(r"Eval(?:uation)?[:\s].*reward[:=]\s*([-\d\.e]+)", txt, re.I)
    if not m:
        m = re.search(r"mean[_\s]?return[:=]\s*([-\d\.e]+)", txt, re.I)
    if m:
        reward = float(m.group(1))
    # cost heuristic
    m2 = re.search(r"cost[:=]\s*([-\d\.e]+)", txt, re.I)
    if m2:
        cost = float(m2.group(1))
    # fallback: look for progress.csv in runs subfolder
    if reward is None:
        run_dir_candidates = [d for d in os.listdir(os.path.join(SAFEPO_ROOT, "runs")) if d.startswith(STUDY_NAME)]
        for d in run_dir_candidates:
            p = os.path.join(SAFEPO_ROOT, "runs", d, "progress.csv")
            if os.path.exists(p):
                try:
                    with open(p, newline='') as cf:
                        reader = csv.DictReader(cf)
                        last = None
                        for row in reader:
                            last = row
                        if last:
                            for key in ["eval_reward","eval_return","mean_reward","mean_return"]:
                                if key in last:
                                    reward = float(last[key]); break
                            for key in ["eval_cost","mean_cost","cost"]:
                                if key in last:
                                    cost = float(last[key]); break
                except:
                    pass
    # final fallbacks
    if reward is None: reward = -1e9
    if cost is None: cost = 1e9
    return reward, cost

def make_cmd(run_dir, seed, params):
    """Build CLI command to run macpo.py. Adjust flags to match macpo.py if needed."""
    cmd = [
        "python", TRAIN_SCRIPT,
        "--task", TASK,
        "--seed", str(seed),
        "--experiment", "optuna",
        "--write-terminal", "False",
        "--headless", "True",
        "--total-steps", str(TOTAL_STEPS_PER_TRIAL),
        "--log-dir", run_dir   # try this; if macpo.py uses a different flag, change below
    ]
    # mapping of param names -> CLI flags (**check macpo.py and adapt these flag names**)
    flag_map = {
        "pi_lr": "--pi-lr",
        "vf_lr": "--vf-lr",
        "clip_ratio": "--clip-ratio",
        "entropy_coef": "--entropy-coef",
        "lam": "--lam",
        "gamma": "--gamma",
        "lagrangian_lr": "--lagrangian-lr",
        "lagrangian_init": "--lagrangian-init"
    }
    for k,v in params.items():
        if k in flag_map:
            cmd += [flag_map[k], str(v)]
    return cmd

def objective(trial):
    # search space (practical starting ranges)
    pi_lr = trial.suggest_loguniform("pi_lr", 3e-5, 3e-3)
    vf_lr = trial.suggest_loguniform("vf_lr", 1e-5, 1e-3)
    clip_ratio = trial.suggest_uniform("clip_ratio", 0.1, 0.4)
    entropy_coef = trial.suggest_uniform("entropy_coef", 0.0, 0.02)
    lam = trial.suggest_uniform("lam", 0.90, 0.99)
    gamma = trial.suggest_uniform("gamma", 0.95, 0.999)
    lagrangian_lr = trial.suggest_loguniform("lagrangian_lr", 1e-4, 1e-1)
    lagrangian_init = trial.suggest_loguniform("lagrangian_init", 1e-4, 1.0)

    params = dict(pi_lr=pi_lr, vf_lr=vf_lr, clip_ratio=clip_ratio,
                  entropy_coef=entropy_coef, lam=lam, gamma=gamma,
                  lagrangian_lr=lagrangian_lr, lagrangian_init=lagrangian_init)

    trial_id = trial.number
    seed = BASE_SEED + trial_id
    run_dir = os.path.join(SAFEPO_ROOT, "runs", f"{STUDY_NAME}_trial{trial_id}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    cmd = make_cmd(run_dir, seed, params)
    logpath = os.path.join(run_dir, "train.log")
    print("Running trial", trial_id, "cmd:", " ".join(cmd))
    with open(logpath, "wb") as outf:
        p = subprocess.run(cmd, cwd=SAFEPO_ROOT, stdout=outf, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        # failed run, tell Optuna to skip
        raise optuna.exceptions.TrialPruned()

    reward, cost = parse_metric_from_log(logpath)
    # constraint-aware objective
    penalty = max(0.0, cost - COST_LIMIT)
    score = reward - BETA * penalty
    # report to optuna
    trial.set_user_attr("reward", reward)
    trial.set_user_attr("cost", cost)
    trial.set_user_attr("params", params)
    print(f"Trial {trial_id} -> reward={reward}, cost={cost}, score={score}")
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--jobs", type=int, default=N_JOBS)
    parser.add_argument("--study", type=str, default=STUDY_NAME)
    args = parser.parse_args()

    study = optuna.create_study(study_name=args.study, direction="maximize", load_if_exists=True)
    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs)
    print("Best trial:", study.best_trial.number, study.best_trial.value)
    print(study.best_trial.user_attrs)

if __name__ == "__main__":
    main()
