import os
import glob

import optuna


def merge_hpo_databases(storage_dir, study_name, output_db="hpo_final.db"):
    """
    Merges multiple SQLite Optuna databases into one to find the global best parameters.
    """
    # 1. Find all DBs
    db_files = glob.glob(os.path.join(storage_dir, "*.db"))
    if not db_files:
        print("No databases found.")
        return

    print(f"Found {len(db_files)} databases. Merging into {output_db}...")

    # 2. Create Target Study
    if output_db.startswith('sqlite:///'):
        raise RuntimeError(f"'output_db' must be a path to file, got {output_db}...")
    target_storage = f"sqlite:///{output_db}"

    # Delete if exists to start fresh merge
    if os.path.exists(output_db):
        os.remove(output_db)

    target_study = optuna.create_study(study_name=study_name, storage=target_storage,
                                       direction='minimize')

    # 3. Smart Merging with Conflict Resolution
    seen_signatures = set()
    total_added = 0
    total_skipped = 0

    # Iterate and Copy Trials
    for db in db_files:
        print(f"Merging {db}...")
        try:
            source_storage = f"sqlite:///{db}"
            source_study = optuna.load_study(study_name=study_name,
                                             storage=source_storage)

            for trial in source_study.trials:
                # Only keep successful trials
                if trial.state != optuna.trial.TrialState.COMPLETE:
                    continue

                # Create a unique, hashable signature from the hyperparameters
                # e.g., (('l_grad', 0.5), ('lr', 0.001), ...)
                param_signature = tuple(sorted(trial.params.items()))

                if param_signature in seen_signatures:
                    total_skipped += 1
                    continue # skip duplicate!

                # Add to set and insert into target database
                seen_signatures.add(param_signature)
                target_study.add_trial(trial)
                total_added += 1
        except Exception as e:
            print(f"Failed to merge {db}: {e}")

    print(f"Merge Complete. Total Trials: {len(target_study.trials)}")
    # print(f"Global Best Params: {target_study.best_params}")
    # print(f"Global Best Value: {target_study.best_value}")
    # print("\n--- Merge Complete ---")
    print(f"Unique Trials Added: {total_added}")
    print(f"Duplicate Trials Skipped: {total_skipped}")
    print(f"Global Best Params: {target_study.best_params}")
    print(f"Global Best Value: {target_study.best_value}")
