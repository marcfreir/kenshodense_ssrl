"""
pretrain.py
Self-supervised pretraining of KenshoDense on the F3 seismic dataset, once
per augmentation configuration (see transformations.py) and per repeated
run.

Each (configuration, run) pair produces two checkpoints in
``<output_root>/checkpoints/pretrain/rn50x{k}/run{r}/``:

    kenshodense_<config>_run<r>_<epochs>epochs_best.ckpt   (lowest val_loss)
    kenshodense_<config>_run<r>_<epochs>epochs_last.ckpt   (final epoch)

Runs whose checkpoint already exists are skipped, so an interrupted job can
simply be relaunched. The seed of every run depends only on the run index
(see seed_utils.py), which makes run r of any two configurations a valid
pair for paired statistical tests.

Examples
--------
    # all 26 configurations, run 1, RN50 4x on GPUs 0 and 1
    python pretrain.py --runs 1 --width 4 --devices 0 1

    # six repeated runs of the Gaussian Blur and Translation families
    python pretrain.py --runs 1 2 3 4 5 6 --families GB Tr

    # list the available configurations
    python pretrain.py --list
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.data.data_modules.f3 import F3DataModule

from kenshodense import KenshoDense
from deeplabv3 import DeepLabV3Backbone
from transformations import get_transform_dict, describe_grid, FAMILIES
from seed_utils import seed_everything_for_run
import paths

# ============================================================================
# PRETRAINING HYPERPARAMETERS
# ============================================================================
# Used both to build the model and to record the run metadata, so the two
# can never drift apart.
PRETRAIN_HPARAMS = {
    "batch_size": 16,
    "num_workers": 2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "optimizer_betas": (0.9, 0.95),
    "optimizer_eps": 1e-6,
    "steps_per_epoch": 100,
}


# ============================================================================
# METADATA / RESULTS SAVING HELPERS
# ============================================================================

def flatten_dict(d, parent_key='', sep='_'):
    """Flatten a nested dictionary for CSV export."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if len(v) <= 10 and all(isinstance(x, (int, float)) for x in v):
                for i, item in enumerate(v):
                    items.append((f"{new_key}_{i}", item))
            else:
                items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def save_experiment_metadata(experiment_data, save_dir, filename_prefix):
    """Save experiment metadata and parameters to JSON and CSV files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_file = save_dir / f"{filename_prefix}_metadata_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(experiment_data, f, indent=2, default=str)

    csv_file = save_dir / f"{filename_prefix}_metadata_{timestamp}.csv"
    pd.DataFrame([flatten_dict(experiment_data)]).to_csv(csv_file, index=False)

    print("Metadata saved to:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")

    return json_file, csv_file


def save_loss_history(loss_history, save_dir, filename_prefix, run_key):
    """Save per-epoch loss history to JSON and CSV files."""
    if not loss_history:
        print("No loss history to save")
        return None, None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_data = {
        'run_key': run_key,
        'timestamp': timestamp,
        'total_epochs': len(loss_history.get('train_losses', [])),
        'loss_history': loss_history,
    }

    json_file = save_dir / f"{filename_prefix}_{run_key}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(loss_data, f, indent=2, default=str)

    rows = []
    max_epochs = max(len(v) if isinstance(v, list) else 1 for v in loss_history.values())
    for epoch in range(max_epochs):
        row = {'epoch': epoch, 'run_key': run_key}
        for loss_type, loss_values in loss_history.items():
            if isinstance(loss_values, list) and epoch < len(loss_values):
                row[loss_type] = loss_values[epoch]
            elif not isinstance(loss_values, list):
                row[loss_type] = loss_values if epoch == 0 else None
        rows.append(row)

    csv_file = save_dir / f"{filename_prefix}_{run_key}_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)

    print("Loss history saved to:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")

    return json_file, csv_file


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def find_existing_checkpoint(ckpt_dir: Path, model_name: str):
    """Return (path, type) if a checkpoint already exists, else (None, None)."""
    best_path = ckpt_dir / f"{model_name}_best.ckpt"
    last_path = ckpt_dir / f"{model_name}_last.ckpt"
    if best_path.exists():
        return str(best_path), "best"
    if last_path.exists():
        return str(last_path), "last"
    found = list(ckpt_dir.glob(f"{model_name}*.ckpt"))
    if found:
        return str(found[0]), "found"
    return None, None


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_kenshodense(
    config_key: str,
    transformation_pipeline,
    run_idx: int,
    width_multiplier: int = 1,
    epochs: int = 100,
    devices=(0, 1),
    data_root: Path = paths.DEFAULT_DATA_ROOT,
    output_root: Path = paths.DEFAULT_OUTPUT_ROOT,
    save_outputs: bool = True,
):
    """
    Pretrain KenshoDense with a single augmentation configuration.

    Args:
        config_key: Augmentation configuration key (e.g. "GB_param1").
        transformation_pipeline: The augmentation pipeline used to build
            every view.
        run_idx: Repeated-run index (1..R). Controls both the checkpoint
            directory and the seed (seed_utils.seed_everything_for_run).
        width_multiplier: ResNet-50 width (1, 2 or 4).
        epochs: Number of pretraining epochs.
        devices: GPU indices. More than one device uses DDP.
        data_root: Dataset root (contains images/ and annotations/).
        output_root: Root of all outputs (see paths.py).
        save_outputs: Whether to save JSON/CSV metadata and loss history.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    train_path = paths.images_dir(data_root)
    annotation_path = paths.annotations_dir(data_root)
    devices = list(devices)
    hp = PRETRAIN_HPARAMS

    run_key = f"{config_key}_run{run_idx}"
    model_name = paths.pretrain_model_name(config_key, run_idx, epochs)
    ckpt_dir = paths.pretrain_ckpt_dir(output_root, width_multiplier, run_idx)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Check for an existing checkpoint BEFORE building the datamodule,
    # backbone, model or trainer, so a relaunch after a crash skips
    # completed runs without paying for any GPU allocation.
    existing_ckpt_path, existing_ckpt_type = find_existing_checkpoint(ckpt_dir, model_name)
    if existing_ckpt_path is not None:
        print(f"\n⏭️  SKIPPING {run_key} — checkpoint already exists:")
        print(f"   {existing_ckpt_path} ({existing_ckpt_type})")
        return {
            'config_key': config_key,
            'run_idx': run_idx,
            'model_name': model_name,
            'checkpoint_path': existing_ckpt_path,
            'checkpoint_type': existing_ckpt_type,
            'training_success': True,
            'test_success': False,
            'skipped': True,
        }

    # Seed everything BEFORE any randomness (data split, weight init,
    # augmentation RNG state) occurs. The seed depends only on run_idx, so
    # run X is directly comparable across all configurations.
    seed = seed_everything_for_run(run_idx)

    data_module = F3DataModule(
        root_data_dir=str(train_path),
        root_annotation_dir=str(annotation_path),
        train_transforms=None,
        batch_size=hp["batch_size"],
        num_workers=hp["num_workers"],
    )

    backbone = DeepLabV3Backbone(pretrained=False, width_multiplier=width_multiplier)

    print(f"Training configuration: {config_key} (run {run_idx})")
    print(f"Pipeline: {transformation_pipeline}")

    use_gpu = torch.cuda.is_available()
    precision = "16-mixed" if use_gpu else "32"
    strategy = DDPStrategy(find_unused_parameters=True) if len(devices) > 1 else "auto"

    experiment_params = {
        'config_key': config_key,
        'transformation_pipeline': str(transformation_pipeline),
        'epochs': epochs,
        'run_idx': run_idx,
        'seed': seed,
        **hp,
        'train_path': str(train_path),
        'annotation_path': str(annotation_path),
        'backbone': f'DeepLabV3-RN50x{width_multiplier}',
        'width_multiplier': width_multiplier,
        'backbone_out_channels': backbone.out_channels,
        'pretrained_backbone': False,
        'precision': precision,
        'devices': devices,
        'strategy': type(strategy).__name__ if not isinstance(strategy, str) else strategy,
    }

    kenshodense_model = KenshoDense(
        backbone=backbone,
        augmentations=transformation_pipeline,
        optimizer_name=hp["optimizer"],
        optimizer_params={"betas": hp["optimizer_betas"], "eps": hp["optimizer_eps"]},
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        max_epochs=epochs,
        steps_per_epoch=hp["steps_per_epoch"],
        batch_size=hp["batch_size"],
    )

    outputs_dir = paths.results_dir(output_root, "pretrain", width_multiplier)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = paths.logs_dir(output_root, "pretrain")
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(
        save_dir=logs_dir,
        name=f"kenshodense_{paths.width_tag(width_multiplier)}_{config_key}",
        version=f"run{run_idx}_{epochs}epochs",
    )

    best_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{model_name}_best",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
    )

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{model_name}_last",
        save_top_k=1,
        monitor="epoch",
        mode="max",
        save_weights_only=True,
        save_last=False,
    )

    # deterministic=True is intentionally NOT set: several ops used by
    # KenshoDense / the seismic augmentations have no deterministic CUDA
    # kernels. Reproducibility comes from explicit seeding instead; the
    # small residual kernel-level nondeterminism is one of the reasons the
    # study uses repeated runs.
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices if use_gpu else 1,
        precision=precision,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
        strategy=strategy,
        log_every_n_steps=10,
        logger=csv_logger,
    )

    pipeline = SimpleLightningPipeline(
        model=kenshodense_model,
        trainer=trainer,
        save_run_status=True,
    )

    print(f"Starting training for {run_key}...")

    training_start_time = datetime.now()
    try:
        pipeline.run(data=data_module, task="fit")
        training_success = True
        training_end_time = datetime.now()
        training_duration = (training_end_time - training_start_time).total_seconds()

        print(f"Training completed! CSV logs saved to: {csv_logger.log_dir}")

    except Exception as e:
        print(f"Training failed for {run_key}: {str(e)}")
        training_end_time = datetime.now()
        training_duration = (training_end_time - training_start_time).total_seconds()

        result = {
            'config_key': config_key,
            'run_idx': run_idx,
            'model_name': model_name,
            'training_success': False,
            'training_duration_seconds': training_duration,
            'error': str(e),
            'experiment_params': experiment_params,
            'csv_log_dir': str(csv_logger.log_dir) if csv_logger.log_dir else None,
        }

        if save_outputs:
            save_experiment_metadata(result, outputs_dir, f"failed_{run_key}")

        return result

    checkpoint_path, checkpoint_type = find_existing_checkpoint(ckpt_dir, model_name)
    if checkpoint_path is None:
        checkpoint_type = "none"

    test_start_time = datetime.now()
    test_success = False
    test_duration = 0
    if checkpoint_path:
        try:
            pipeline.run(data=data_module, task="test", ckpt_path=checkpoint_path)
            test_success = True
        except Exception as e:
            print(f"Testing failed for {run_key}: {str(e)}")
        test_duration = (datetime.now() - test_start_time).total_seconds()

    loss_history = kenshodense_model.get_loss_history()

    result = {
        'config_key': config_key,
        'run_idx': run_idx,
        'model_name': model_name,
        'checkpoint_path': checkpoint_path,
        'checkpoint_type': checkpoint_type,
        'training_success': training_success,
        'test_success': test_success,
        'skipped': False,
        'training_duration_seconds': training_duration,
        'test_duration_seconds': test_duration,
        'total_duration_seconds': training_duration + test_duration,
        'loss_history': loss_history,
        'experiment_params': experiment_params,
        'training_start_time': training_start_time.isoformat(),
        'training_end_time': training_end_time.isoformat(),
        'final_epoch': epochs,
        'csv_log_dir': str(csv_logger.log_dir),
    }

    if save_outputs:
        save_experiment_metadata(result, outputs_dir, f"experiment_{run_key}")
        if loss_history:
            save_loss_history(loss_history, outputs_dir, "losses", run_key)

    return result


# ============================================================================
# STUDY RUNNER
# ============================================================================

def run_study(
    experiments_dict: dict,
    runs,
    width_multiplier: int = 1,
    epochs: int = 100,
    devices=(0, 1),
    data_root: Path = paths.DEFAULT_DATA_ROOT,
    output_root: Path = paths.DEFAULT_OUTPUT_ROOT,
):
    """Pretrain every configuration in ``experiments_dict`` for every run
    index in ``runs`` and save a summary of the study."""
    runs = list(runs)

    print(f"\n{'='*80}")
    print("KENSHODENSE AUGMENTATION STUDY — PRETRAINING")
    print(f"{'='*80}")
    print(f"Backbone: ResNet-50 ({width_multiplier}x width)")
    print(f"Configurations: {len(experiments_dict)}")
    print(f"Run indices: {runs}")
    print(f"Total training runs: {len(experiments_dict) * len(runs)}")
    print(f"Epochs per run: {epochs}")

    all_results = {}

    for exp_idx, (config_key, transform_pipeline) in enumerate(experiments_dict.items(), 1):
        print(f"\n{'-'*80}")
        print(f"Configuration {exp_idx}/{len(experiments_dict)}: {config_key}")
        print(f"{'-'*80}")

        exp_results = []

        for run_idx in runs:
            print(f"\nRun {run_idx} for {config_key}")

            try:
                result = train_kenshodense(
                    config_key=config_key,
                    transformation_pipeline=transform_pipeline,
                    run_idx=run_idx,
                    width_multiplier=width_multiplier,
                    epochs=epochs,
                    devices=devices,
                    data_root=data_root,
                    output_root=output_root,
                    save_outputs=True,
                )
                exp_results.append(result)

                if result['training_success']:
                    print(f"✅ Run {run_idx} SUCCESS")
                else:
                    print(f"❌ Run {run_idx} FAILED: {result.get('error', 'Unknown')}")

            except Exception as e:
                print(f"❌ Run {run_idx} EXCEPTION: {str(e)}")
                exp_results.append({
                    'training_success': False,
                    'error': str(e),
                    'run_idx': run_idx,
                })

        all_results[config_key] = exp_results

    summary_dir = paths.results_dir(output_root, "pretrain", width_multiplier)
    save_study_summary(all_results, summary_dir)

    print(f"\n{'='*80}")
    print("PRETRAINING COMPLETE!")
    print(f"{'='*80}")

    return all_results


def save_study_summary(results_dict, summary_dir):
    """Save per-configuration summary statistics of the study."""
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_data = []
    for config_key, exp_results in results_dict.items():
        successful_runs = [r for r in exp_results if r.get('training_success', False)]

        summary_row = {
            'config_key': config_key,
            'total_runs': len(exp_results),
            'successful_runs': len(successful_runs),
            'failed_runs': len(exp_results) - len(successful_runs),
            'success_rate': len(successful_runs) / len(exp_results) if exp_results else 0,
        }

        if successful_runs:
            durations = [r.get('training_duration_seconds', 0) for r in successful_runs]
            summary_row['avg_duration_seconds'] = sum(durations) / len(durations)
            summary_row['min_duration_seconds'] = min(durations)
            summary_row['max_duration_seconds'] = max(durations)

        summary_data.append(summary_row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = summary_dir / f"pretrain_summary_{timestamp}.csv"
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)

    json_file = summary_dir / f"pretrain_detailed_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)

    print("\nStudy summary saved to:")
    print(f"  CSV: {summary_file}")
    print(f"  JSON: {json_file}")


# ============================================================================
# COMMAND LINE
# ============================================================================

def select_experiments(experiments: dict, families=None, configs=None) -> dict:
    """Filter the experiment dict by family and/or explicit config keys."""
    if configs:
        unknown = sorted(set(configs) - set(experiments))
        if unknown:
            raise SystemExit(f"Unknown configuration(s): {unknown}. Use --list to see them.")
        return {k: experiments[k] for k in configs}
    if families:
        return {k: v for k, v in experiments.items() if k.split("_")[0] in families}
    return experiments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain KenshoDense with each augmentation configuration.")
    parser.add_argument("--runs", type=int, nargs="+", default=[1],
                        help="Repeated-run indices to train (default: 1). "
                             "The seed of each run is SEED_BASE + run index.")
    parser.add_argument("--width", type=int, choices=(1, 2, 4), default=1,
                        help="ResNet-50 width multiplier (default: 1).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1],
                        help="GPU indices; more than one uses DDP (default: 0 1).")
    parser.add_argument("--families", nargs="+", choices=FAMILIES,
                        help="Only train these augmentation families.")
    parser.add_argument("--configs", nargs="+",
                        help="Only train these configurations (e.g. GB_param1 FF_param3).")
    parser.add_argument("--data-root", type=Path, default=paths.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=paths.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--list", action="store_true",
                        help="List the augmentation configurations and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    experiments = get_transform_dict(img_size=paths.IMG_SIZE)

    if args.list:
        for key, params in describe_grid().items():
            print(f"{key:<12} {params}")
        raise SystemExit(0)

    experiments = select_experiments(experiments, args.families, args.configs)

    run_study(
        experiments_dict=experiments,
        runs=args.runs,
        width_multiplier=args.width,
        epochs=args.epochs,
        devices=args.devices,
        data_root=args.data_root,
        output_root=args.output_root,
    )
