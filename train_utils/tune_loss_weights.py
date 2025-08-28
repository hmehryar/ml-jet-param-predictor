# tools/tune_loss_weights.py
from __future__ import annotations
import argparse, json, os
from types import SimpleNamespace

# 🔧 Add root to sys.path for module imports (Jupyter-safe)
import sys
from pathlib import Path


ROOT = Path.cwd()  # Go up from /notebooks/ → experiment → project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

print(f"[INFO] Added ROOT to sys.path: {ROOT}")



import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# import your existing training entrypoint & config loader
# from train_utils.training_loop import run_training_loop
# after:
from train_utils.training_loop import run_training_with_config
from config import get_config  # adjust if your loader name differs

def make_cfg_with_lambdas(base_cfg: SimpleNamespace, l1: float, l2: float, l3: float,
                          *, max_epochs_override: int | None = None) -> SimpleNamespace:
    # clone cfg (shallow) and override loss weights
    cfg = SimpleNamespace(**vars(base_cfg))
    lw = dict(getattr(cfg, "loss_weights", {}) or {})
    lw["energy_loss_output"] = float(l1)
    lw["alpha_output"]        = float(l2)
    lw["q0_output"]           = float(l3)
    cfg.loss_weights = lw

    # optional: shorten epochs for faster HPO
    if max_epochs_override is not None:
        cfg.epochs = int(max_epochs_override)
        # Optional: tighten patience if present
        if hasattr(cfg, "patience"):
            cfg.patience = max(3, min(cfg.epochs // 4, getattr(cfg, "patience")))
    return cfg


def build_objective(base_cfg: SimpleNamespace,
                    *, guardrail_drop: float = 0.03,
                    baseline_path: str | None = None,
                    hpo_epochs: int | None = None):
    """
    guardrail_drop: max allowed fractional drop vs baseline (e.g., 0.03 = 3%) for energy and alpha acc
    baseline_path: optional path to a JSON file with baseline metrics to guard against (λ=1,1,1)
                   format: {"energy_acc": ..., "alpha_acc": ...}
    hpo_epochs: override number of epochs per trial (shorter = faster)
    """
    baseline_energy_acc = None
    baseline_alpha_acc  = None
    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            b = json.load(f)
        baseline_energy_acc = float(b.get("energy_acc", 0.0))
        baseline_alpha_acc  = float(b.get("alpha_acc", 0.0))

    def objective(trial: optuna.trial.Trial) -> float:
        # search space (adjust if you want)
        l1 = trial.suggest_float("lambda_energy", 0.25, 1.00)  # energy
        l2 = trial.suggest_float("lambda_alpha",  0.75, 1.50)  # alpha_s
        l3 = trial.suggest_float("lambda_q0",     1.25, 3.00)  # Q0

        # build cfg with overrides + epoch shortening for HPO
        cfg = make_cfg_with_lambdas(base_cfg, l1, l2, l3, max_epochs_override=hpo_epochs)

        # Important: give each trial a distinct tag/output dir suffix
        tag_suffix = f"lam_{l1:.2f}_{l2:.2f}_{l3:.2f}"
        if hasattr(cfg, "model_tag"):
            cfg.model_tag = f"{cfg.model_tag}_HPO_{tag_suffix}"
        else:
            cfg.model_tag = f"HPO_{tag_suffix}"

        # Run training → returns metrics dict (ensure training_loop returns validation metrics)
        # metrics = run_training_loop(cfg)  # must return metrics like in your evaluate()
        best_metric = run_training_with_config(cfg)
        

        q0_f1    = float(best_metric["q0"]["f1"])
        alpha_acc = float(best_metric["alpha"]["accuracy"])
        energy_acc = float(best_metric["energy"]["accuracy"])

        # Guardrails vs baseline if provided (allow ≤ guardrail_drop drop)
        if baseline_energy_acc is not None and energy_acc < baseline_energy_acc * (1.0 - guardrail_drop):
            # Penalize heavily
            return 0.0
        if baseline_alpha_acc is not None and alpha_acc < baseline_alpha_acc * (1.0 - guardrail_drop):
            return 0.0

        # Report the target to Optuna
        trial.set_user_attr("alpha_acc", alpha_acc)
        trial.set_user_attr("energy_acc", energy_acc)
        trial.set_user_attr("q0_acc", float(best_metric["q0"]["accuracy"]))
        trial.set_user_attr("model_tag", cfg.model_tag)

        # If you’d like pruning mid-training, wire per-epoch callbacks in run_training_loop
        # and call `trial.report(q0_f1_epoch, epoch)` + `trial.should_prune()`.
        return q0_f1

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    ap.add_argument("--hpo-epochs", type=int, default=8, help="Epochs per trial for speed")
    ap.add_argument("--study-name", default="tune_loss_weights", help="Optuna study name")
    ap.add_argument("--storage", default="sqlite:///optuna_loss_weights.db", help="Optuna storage URI")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--baseline-json", default="", help="Optional JSON with baseline energy/alpha acc")
    args = ap.parse_args()

    # Load your base cfg from YAML
    base_cfg = get_config(args.config)

    # Fix Optuna randomness for reproducibility
    sampler = TPESampler(seed=args.seed, multivariate=True, group=True)
    pruner = MedianPruner(n_startup_trials=max(5, args.trials // 5), n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    obj = build_objective(
        base_cfg,
        guardrail_drop=0.03,
        baseline_path=args.baseline_json or None,
        hpo_epochs=args.hpo_epochs
    )

    study.optimize(obj, n_trials=args.trials, show_progress_bar=True)

    print("\n=== Best Trial ===")
    bt = study.best_trial
    print("Value (Q0 F1):", bt.value)
    print("Params:", bt.params)
    print("Attrs:", bt.user_attrs)

    # Save best params to a JSON for future training
    best = {
        "lambda_energy": bt.params["lambda_energy"],
        "lambda_alpha":  bt.params["lambda_alpha"],
        "lambda_q0":     bt.params["lambda_q0"],
        "q0_f1":         bt.value,
        "alpha_acc":     bt.user_attrs.get("alpha_acc", None),
        "energy_acc":    bt.user_attrs.get("energy_acc", None),
        "q0_acc":        bt.user_attrs.get("q0_acc", None),
        "model_tag":     bt.user_attrs.get("model_tag", None),
    }
    os.makedirs("optuna_artifacts", exist_ok=True)
    with open(os.path.join("optuna_artifacts", f"{args.study_name}_best.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("Wrote:", f.name)

if __name__ == "__main__":
    main()
