#!/usr/bin/env python3
"""
Script to check the best results from the parallel Optuna study
"""
import optuna
import os

def check_study_results():
    study_name = "macpo_hyperopt"
    storage_url = "sqlite:///optuna_study.db"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        print(f"=== Optuna Study Results ===")
        print(f"Study name: {study_name}")
        print(f"Number of trials: {len(study.trials)}")
        print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"Number of running trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])}")
        
        if study.best_trial:
            print(f"\n=== BEST TRIAL, {study.best_trial.number} ===")
            print(f"Best value (reward): {study.best_value:.4f}")
            print(f"Best parameters:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
            print(f"Trial number: {study.best_trial.number}")
        else:
            print("No completed trials yet.")
            
        # Show recent trials
        recent_trials = sorted(study.trials, key=lambda t: t.number)[-5:]
        print(f"\n=== RECENT 5 TRIALS ===")
        for trial in recent_trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                print(f"Trial {trial.number}: {trial.value:.4f} - {trial.state}")
            else:
                print(f"Trial {trial.number}: {trial.state}")
    
    except Exception as e:
        print(f"Could not load study: {e}")
        print("Make sure the database file exists and contains the study.")

if __name__ == "__main__":
    check_study_results()