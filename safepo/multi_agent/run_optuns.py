import optuna
import subprocess
import sys
import os
import re
import glob
import pandas as pd

def get_trial_results(trial_number):
    runs_dir = "safepo/runs"
    pattern = f"{runs_dir}/**/seed-{trial_number:03d}-*/progress.csv"
    csv_files = glob.glob(pattern, recursive=True)
    
    if not csv_files:
        return None
    
    df = pd.read_csv(csv_files[0])
    final_rows = df.iloc[-10:]
    avg_reward = final_rows['Metrics/EpRet'].mean()
    avg_cost = final_rows['Metrics/EpCost'].mean()
    
    return avg_reward, avg_cost

def objective(trial):
    # === Suggest hyperparameters for this trial ===
    actor_lr      = trial.suggest_float("actor_lr", 3e-5, 3e-3, log=True)
    critic_lr     = trial.suggest_float("critic_lr", 1e-4, 3e-3, log=True)
    entropy_coef  = trial.suggest_float("entropy_coef", 0.001, 0.05)
    clip_param    = trial.suggest_float("clip_param", 0.1, 0.3)
    learning_iters= trial.suggest_int("learning_iters", 10, 20)
    safety_gamma  = trial.suggest_float("safety_gamma", 0.01, 0.5)
    target_kl     = trial.suggest_float("target_kl", 0.005, 0.02)
    # cost_limit    = trial.suggest_float("cost_limit", 10.0, 50.0)
    cost_limit = 0
    # === Set environment variables if needed by your script ===
    env = os.environ.copy()
    env.update({
        'OPTUNA_ACTOR_LR': str(actor_lr),
        'OPTUNA_CRITIC_LR': str(critic_lr),
        'OPTUNA_ENTROPY_COEF': str(entropy_coef),
        'OPTUNA_CLIP_PARAM': str(clip_param),
        'OPTUNA_LEARNING_ITERS': str(learning_iters),
        'OPTUNA_SAFETY_GAMMA': str(safety_gamma),
        'OPTUNA_TARGET_KL': str(target_kl),
        # 'OPTUNA_COST_LIMIT': str(cost_limit)
    })

    # === Locate macpo.py script ===
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "macpo.py"))

    # if not os.path.exists(script_path):
    #     script_path = os.path.join(os.getcwd(), "safepo", "multi_agent", "macpo.py")
    
    # === Build command ===
    cmd = [
        sys.executable, script_path,
        "--task", "8m",
        "--total-steps", "3000",
        "--num-envs", "3", 
        "--cost-type", "danger_zone",
        "--seed", str(trial.number),
        "--cost-limit", str(cost_limit),
        "--write-terminal", "True",
        "--actor-lr", str(actor_lr),
        "--critic-lr", str(critic_lr),
        "--entropy-coef", str(entropy_coef),
        "--clip-param", str(clip_param),
        "--learning-iters", str(learning_iters),
        "--safety-gamma", str(safety_gamma),
        "--target-kl", str(target_kl)
    ]
    
    print(f"Trial {trial.number}: actor_lr={actor_lr}, critic_lr={critic_lr}, "
          f"entropy={entropy_coef}, clip={clip_param}, learning_iters={learning_iters}, "
          f"safety_gamma={safety_gamma}, target_kl={target_kl}, cost_limit={cost_limit}")
    
    subprocess.run(cmd, text=True, env=env, cwd=os.path.dirname(script_path))

    # === Retrieve reward and cost from your logging system ===
    reward, cost = get_trial_results(trial.number)
    if reward is None:
        raise optuna.exceptions.TrialPruned()
    
    # === Penalize if constraint is violated ===
    if cost > cost_limit:
        violation_ratio = (cost - cost_limit) / cost_limit
        return reward * (1 - violation_ratio)

    return reward



if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    if study.trials:
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            print("Best trial:")
            print(study.best_params)
            print(f"Best value: {study.best_value}")
        else:
            print("No trials completed successfully")
    else:
        print("No trials run")