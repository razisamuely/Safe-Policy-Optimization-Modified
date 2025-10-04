#!/usr/bin/env python3
import optuna

def print_all_trials():
    study_name = "macpo_hyperopt"
    storage_url = "sqlite:///optuna_study.db"
    
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    print(f"=== ALL TRIALS ({len(study.trials)} total) ===")
    for trial in study.trials:
        print(f"Trial {trial.number}: Value={trial.value}, State={trial.state.name}")
        print(f"  Params: {trial.params}")
        print()
    
    if study.best_trial:
        print(f"=== BEST TRIAL ===")
        print(f"Best: Trial {study.best_trial.number} = {study.best_value}")

if __name__ == "__main__":
    print_all_trials()