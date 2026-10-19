"""
finetune.py
Supervised finetuning of KenshoDense-pretrained backbones for seismic facies
segmentation (DeepLabV3 head) on the F3 dataset.

For every pretrain run index, this script discovers the KenshoDense
checkpoints written by pretrain.py in
``<output_root>/checkpoints/pretrain/rn50x{k}/run{r}/``, extracts the online
backbone, attaches a DeepLabV3 head and finetunes the whole network. Each
source checkpoint produces two finetuned checkpoints in
``<output_root>/checkpoints/finetune/rn50x{k}/run{r}/``:

    <config>_run<r>_best_from_<src>.ckpt   (lowest val_loss)
    <config>_run<r>_last_from_<src>.ckpt   (final epoch)

where <src> is the pretrain checkpoint used (``last`` by default).
Finetuning is seeded with the same run-index seed as pretraining, so the
paired-run design holds end to end.

Examples
--------
    # finetune all pretrain checkpoints of runs 1..6 (RN50 4x)
    python finetune.py --runs 1 2 3 4 5 6 --width 4

    # only the Frequency Filter family of run 2
    python finetune.py --runs 2 --families FF
"""

import argparse
import gc
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from minerva.data.data_modules.f3 import F3DataModule

from deeplabv3 import DeepLabV3, DeepLabV3Backbone
from transformations import FAMILIES
from seed_utils import seed_everything_for_run
import paths

MODEL_NAME = "kenshodense"

# Matches the checkpoint stems produced by pretrain.py / paths.pretrain_model_name:
#   kenshodense_<config>_run<r>_<epochs>epochs_<best|last>
CKPT_PATTERN = re.compile(
    rf'^{MODEL_NAME}_(?P<config>.+)_run(?P<run>\d+)_(?P<epochs>\d+)epochs_(?P<ctype>best|last)$'
)


# ============================================================================
# CHECKPOINT LOADING / DISCOVERY
# ============================================================================

def load_ssl_backbone_weights(checkpoint_path: str, expected_width: int = None) -> DeepLabV3Backbone:
    """Load the online backbone from a pretrained KenshoDense checkpoint.

    The backbone width (RN50 1x/2x/4x) is inferred from the checkpoint
    itself, so the rebuilt backbone always matches what was pretrained.
    """
    print(f"Loading SSL model weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # KenshoDense stores the online encoder as `backbone_online`. The EMA
    # target networks live under `backbone_targets.*`, which does not match
    # this prefix.
    prefix = "backbone_online."
    backbone_state_dict = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }

    if not backbone_state_dict:
        raise RuntimeError(
            f"No keys matched prefix '{prefix}' in {checkpoint_path}. "
            f"Available top-level prefixes: "
            f"{sorted({k.split('.')[0] for k in state_dict.keys()})}"
        )

    # The stem conv has 64 * k filters.
    stem_key = "RN50model.conv1.weight"
    if stem_key not in backbone_state_dict:
        raise RuntimeError(f"'{stem_key}' not found in {checkpoint_path}; "
                           f"cannot infer the backbone width.")
    width_multiplier = backbone_state_dict[stem_key].shape[0] // 64
    if expected_width is not None and width_multiplier != expected_width:
        print(f"⚠️  Warning: checkpoint is RN50 {width_multiplier}x but "
              f"--width={expected_width}; using {width_multiplier}x "
              f"(from the checkpoint).")
    print(f"Backbone width: RN50 {width_multiplier}x "
          f"({2048 * width_multiplier}-d features)")

    backbone = DeepLabV3Backbone(pretrained=False, width_multiplier=width_multiplier)
    missing_keys, unexpected_keys = backbone.load_state_dict(
        backbone_state_dict, strict=False
    )

    if missing_keys:
        print(f"Missing keys when loading {MODEL_NAME} weights: {len(missing_keys)} keys")
        print(f"  e.g. {missing_keys[:5]}")
    if unexpected_keys:
        print(f"Unexpected keys when loading {MODEL_NAME} weights: {len(unexpected_keys)} keys")
        print(f"  e.g. {unexpected_keys[:5]}")

    if len(missing_keys) == len(backbone.state_dict()):
        raise RuntimeError(
            "Every backbone parameter is missing after load — the backbone "
            "architecture probably doesn't match the one used during pretraining."
        )

    print(f"Successfully loaded {MODEL_NAME} pretrained weights "
          f"({len(backbone_state_dict) - len(unexpected_keys)} tensors applied)")
    return backbone


def parse_checkpoint_name(checkpoint_path: str) -> dict:
    """Extract configuration / run information from a pretrain checkpoint path."""
    stem = Path(checkpoint_path).stem
    match = CKPT_PATTERN.match(stem)

    if match:
        config_key = match.group('config')
        run_num = match.group('run')
        checkpoint_type = match.group('ctype')
    else:
        print(f"⚠️  Warning: filename did not match expected pattern: {stem}")
        config_key = stem[len(MODEL_NAME) + 1:] if stem.startswith(f"{MODEL_NAME}_") else stem
        run_num = None
        checkpoint_type = 'last' if stem.endswith('_last') else 'best'

    return {
        'config_key': config_key,
        'family': config_key.split('_')[0],
        'run': run_num if run_num else '0',
        'full_key': f"{config_key}_run{run_num}" if run_num else config_key,
        'checkpoint_type': checkpoint_type,
        'original_filename': Path(checkpoint_path).name,
    }


def checkpoint_already_processed(info: dict, source_type: str, save_directory: Path) -> bool:
    """True if both finetuned checkpoints already exist for this source checkpoint."""
    full_key = info['full_key']
    best_path = save_directory / f"{full_key}_best_from_{source_type}.ckpt"
    last_path = save_directory / f"{full_key}_last_from_{source_type}.ckpt"

    both_exist = best_path.exists() and last_path.exists()
    if both_exist:
        print(f"  ✓ Already processed: {full_key} (from {source_type})")
    return both_exist


def discover_pretrain_checkpoints(checkpoints_dir: Path) -> dict:
    """Group the pretrain checkpoints of ONE run directory by configuration.

    Returns ``{full_key: {'info': ..., 'best': Path, 'last': Path}}`` (either
    of 'best' / 'last' may be missing).
    """
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    all_checkpoints = sorted(checkpoints_dir.glob(f"{MODEL_NAME}_*_run*_*epochs_*.ckpt"))
    print(f"Found {len(all_checkpoints)} pretrain checkpoint files in {checkpoints_dir}")

    checkpoint_dict = {}
    for ckpt_path in all_checkpoints:
        info = parse_checkpoint_name(str(ckpt_path))
        entry = checkpoint_dict.setdefault(info['full_key'], {'info': info})
        entry[info['checkpoint_type']] = ckpt_path

    with_both = sum(1 for v in checkpoint_dict.values() if 'best' in v and 'last' in v)
    last_only = sum(1 for v in checkpoint_dict.values() if 'last' in v and 'best' not in v)
    best_only = sum(1 for v in checkpoint_dict.values() if 'best' in v and 'last' not in v)
    print(f"  - Configurations found: {len(checkpoint_dict)}")
    print(f"  - With both best and last: {with_both}")
    print(f"  - With last only: {last_only}")
    print(f"  - With best only: {best_only}")

    return checkpoint_dict


def cleanup_resources():
    """Clean up GPU memory and distributed processes."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# FINETUNING
# ============================================================================

def finetune_single_checkpoint(
    checkpoint_path: str,
    source_type: str,
    info: dict,
    save_directory: Path,
    logs_root: Path,
    data_root: Path,
    width_multiplier: int,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    device: int = 0,
):
    """Finetune one pretrain checkpoint and save its best/last variants.

    Args:
        checkpoint_path: Path to the pretrained KenshoDense checkpoint.
        source_type: "best" or "last" (which pretrain checkpoint is used).
        info: Output of parse_checkpoint_name; its 'run' field drives the
            seed of this finetuning run.
        save_directory: Where the finetuned checkpoints are written.
        logs_root: Root directory of the Lightning CSV logs.
        data_root: Dataset root (contains images/ and annotations/).
        width_multiplier: Expected backbone width (the checkpoint wins).
        epochs, batch_size, learning_rate: Finetuning hyperparameters.
        device: GPU index.
    """
    full_key = info['full_key']
    print(f"\n{'='*60}")
    print(f"Finetuning {source_type} checkpoint: {full_key}")
    print(f"  Configuration: {info['config_key']}")
    print(f"  Run: {info['run']}")
    print(f"{'='*60}")

    # Seed BEFORE the datamodule/model are built, with the same run-index
    # seed pretraining used for this checkpoint.
    try:
        run_idx = int(info['run'])
        seed = seed_everything_for_run(run_idx)
    except (TypeError, ValueError):
        print(f"⚠️  Warning: could not parse run index from info['run']="
              f"{info.get('run')!r}; proceeding WITHOUT a fixed seed.")
        run_idx = None
        seed = None

    data_module = F3DataModule(
        root_data_dir=str(paths.images_dir(data_root)),
        root_annotation_dir=str(paths.annotations_dir(data_root)),
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
    )

    try:
        pretrained_backbone = load_ssl_backbone_weights(checkpoint_path, width_multiplier)
    except Exception as e:
        print(f"❌ Failed to load backbone from {checkpoint_path}: {e}")
        return None

    model = DeepLabV3(
        backbone=pretrained_backbone,
        num_classes=paths.NUM_CLASSES,
        learning_rate=learning_rate,
        pretrained=False,
    )

    save_directory.mkdir(parents=True, exist_ok=True)

    best_checkpoint_callback = ModelCheckpoint(
        dirpath=save_directory,
        filename=f"{full_key}_best_from_{source_type}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
        verbose=True,
    )

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=save_directory,
        filename=f"{full_key}_last_from_{source_type}",
        save_top_k=1,
        monitor="epoch",
        mode="max",
        save_weights_only=False,
        save_last=False,
        verbose=True,
    )

    logs_root.mkdir(parents=True, exist_ok=True)
    use_gpu = torch.cuda.is_available()

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=[device] if use_gpu else 1,
        logger=CSVLogger(
            logs_root,
            name=f"finetune_{paths.width_tag(pretrained_backbone.width_multiplier)}",
            version=f"{full_key}_{source_type}_{epochs}ep",
        ),
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
        log_every_n_steps=10,
        precision="16-mixed" if use_gpu else "32",
        enable_progress_bar=True,
    )

    training_start_time = datetime.now()
    try:
        print("Starting finetuning...")
        trainer.fit(model, data_module)
        training_success = True
        training_duration = (datetime.now() - training_start_time).total_seconds()
        print(f"✅ Training completed in {training_duration:.1f}s ({training_duration/60:.1f}m)")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        training_duration = (datetime.now() - training_start_time).total_seconds()
        cleanup_resources()
        return {
            'info': info,
            'source_checkpoint_type': source_type,
            'source_checkpoint_path': checkpoint_path,
            'training_success': False,
            'error': str(e),
            'training_duration_seconds': training_duration,
            'run_idx': run_idx,
            'seed': seed,
        }

    best_from_path = save_directory / f"{full_key}_best_from_{source_type}.ckpt"
    last_from_path = save_directory / f"{full_key}_last_from_{source_type}.ckpt"

    if not best_from_path.exists():
        print(f"⚠️  Warning: Expected best checkpoint not found: {best_from_path}")
    if not last_from_path.exists():
        print(f"⚠️  Warning: Expected last checkpoint not found: {last_from_path}")

    test_metrics = None
    if best_from_path.exists():
        try:
            print("Running test evaluation...")
            data_module.setup("test")
            test_results = trainer.test(model, data_module, ckpt_path=str(best_from_path), verbose=True)
            if test_results:
                test_metrics = test_results[0]
                print("✅ Test evaluation completed")
        except Exception as e:
            print(f"⚠️  Test evaluation failed: {e}")

    cleanup_resources()

    return {
        'info': info,
        'source_checkpoint_type': source_type,
        'source_checkpoint_path': checkpoint_path,
        'training_success': training_success,
        'training_duration_seconds': training_duration,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'run_idx': run_idx,
        'seed': seed,
        'width_multiplier': pretrained_backbone.width_multiplier,
        'best_from_checkpoint_path': str(best_from_path) if best_from_path.exists() else None,
        'last_from_checkpoint_path': str(last_from_path) if last_from_path.exists() else None,
        'test_metrics': test_metrics,
        'csv_log_dir': trainer.logger.log_dir if trainer.logger else None,
    }


def finetune_run(
    run_idx: int,
    width_multiplier: int,
    data_root: Path,
    output_root: Path,
    sources=("last",),
    finetune_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    device: int = 0,
    max_experiments: int = None,
    skip_completed: bool = True,
    families=None,
    configs=None,
):
    """Finetune every pretrain checkpoint of ONE run index.

    Args:
        sources: Which pretrain checkpoints to finetune from: ("last",),
            ("best",) or ("best", "last").
        families / configs: Optional filters on augmentation family
            (e.g. ["GB", "FF"]) or configuration key (e.g. ["GB_param1"]).
    """
    pretrain_dir = paths.pretrain_ckpt_dir(output_root, width_multiplier, run_idx)
    save_dir = paths.finetune_ckpt_dir(output_root, width_multiplier, run_idx)
    logs_root = paths.logs_dir(output_root, "finetune")
    results_root = paths.results_dir(output_root, "finetune", width_multiplier)

    print(f"\n{'#'*80}")
    print(f"# RUN INDEX {run_idx}")
    print(f"#   pretrain checkpoints: {pretrain_dir}")
    print(f"#   finetuned checkpoints: {save_dir}")
    print(f"{'#'*80}")

    try:
        available = discover_pretrain_checkpoints(pretrain_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return {}

    if not available:
        print(f"❌ No checkpoints found for run{run_idx}!")
        print(f"Expected pattern: {MODEL_NAME}_<config>_run{run_idx}_<epochs>epochs_<best|last>.ckpt")
        return {}

    available = {k: v for k, v in available.items() if any(s in v for s in sources)}

    if families:
        available = {k: v for k, v in available.items() if v['info']['family'] in families}
        print(f"Filtered to families {families}: {len(available)} configurations")

    if configs:
        available = {k: v for k, v in available.items() if v['info']['config_key'] in configs}
        print(f"Filtered to configurations {configs}: {len(available)} configurations")

    if max_experiments and len(available) > max_experiments:
        available = dict(list(available.items())[:max_experiments])
        print(f"Limited to first {max_experiments} configurations")

    print(f"\nCheckpoints to process: {len(available)} (from {', '.join(sources)})")
    print(f"Each source checkpoint generates 2 files: *_best_from_<src>.ckpt and *_last_from_<src>.ckpt")
    print(f"Skip completed: {'ENABLED' if skip_completed else 'DISABLED'}\n")

    all_results = {}
    successful = failed = skipped = 0
    suite_start_time = datetime.now()

    for i, (full_key, checkpoint_data) in enumerate(available.items(), 1):
        info = checkpoint_data['info']

        print(f"\n{'='*80}")
        print(f"CHECKPOINT {i}/{len(available)}: {full_key}")
        print(f"{'='*80}")

        checkpoint_results = {'info': info}

        for source in sources:
            if source not in checkpoint_data:
                continue

            print(f"\n--- Finetuning from {source.upper()} pretrain checkpoint ---")

            if skip_completed and checkpoint_already_processed(info, source, save_dir):
                checkpoint_results[f'finetune_from_{source}'] = {
                    'training_success': True,
                    'skipped': True,
                    'best_from_checkpoint_path': str(save_dir / f"{full_key}_best_from_{source}.ckpt"),
                    'last_from_checkpoint_path': str(save_dir / f"{full_key}_last_from_{source}.ckpt"),
                }
                skipped += 1
                continue

            try:
                result = finetune_single_checkpoint(
                    checkpoint_path=str(checkpoint_data[source]),
                    source_type=source,
                    info=info,
                    save_directory=save_dir,
                    logs_root=logs_root,
                    data_root=data_root,
                    width_multiplier=width_multiplier,
                    epochs=finetune_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    device=device,
                )
                checkpoint_results[f'finetune_from_{source}'] = result
                if result and result.get('training_success', False):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Error finetuning from {source.upper()}: {e}")
                checkpoint_results[f'finetune_from_{source}'] = {
                    'training_success': False,
                    'error': str(e),
                }
                failed += 1

        all_results[full_key] = checkpoint_results

        elapsed = (datetime.now() - suite_start_time).total_seconds()
        remaining = (len(available) - i) * (elapsed / i)
        print(f"\nProgress: {i}/{len(available)} | Successful: {successful} | "
              f"Failed: {failed} | Skipped: {skipped} | "
              f"Elapsed: {elapsed/3600:.1f}h | ETA: {remaining/3600:.1f}h")

    total_duration = (datetime.now() - suite_start_time).total_seconds()

    print(f"\n{'='*80}")
    print(f"FINETUNING COMPLETE (run{run_idx})")
    print(f"{'='*80}")
    print(f"Successful: {successful} | Failed: {failed} | Skipped: {skipped}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/3600:.1f}h)")

    generated = list(save_dir.glob("*_from_*.ckpt")) if save_dir.exists() else []
    expected = len(available) * 2 * len(sources)
    print(f"Generated checkpoint files: {len(generated)} (expected up to {expected})")

    save_run_summary(
        all_results, results_root, run_idx, sources,
        extra={
            'width_multiplier': width_multiplier,
            'total_checkpoints': len(available),
            'successful_finetuning_runs': successful,
            'failed_finetuning_runs': failed,
            'skipped_runs': skipped,
            'total_duration_seconds': total_duration,
            'finetune_epochs': finetune_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'generated_checkpoints': len(generated),
            'checkpoint_directory': str(save_dir),
            'families': families,
            'configs': configs,
        },
    )

    return all_results


def save_run_summary(all_results: dict, results_root: Path, run_idx: int, sources, extra: dict):
    """Write the JSON and CSV summaries of one run index's finetuning."""
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_file = results_root / f"finetune_results_run{run_idx}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(
            {
                'experiment_info': {'run_idx': run_idx, 'sources': list(sources),
                                    **extra, 'timestamp': datetime.now().isoformat()},
                'results_by_checkpoint': all_results,
            },
            f, indent=2, default=str,
        )
    print(f"✅ Results saved to: {json_file}")

    rows = []
    for full_key, results in all_results.items():
        info = results['info']
        row = {
            'full_key': full_key,
            'config_key': info['config_key'],
            'family': info['family'],
            'run': info['run'],
        }
        for source in sources:
            r = results.get(f'finetune_from_{source}') or {}
            row[f'from_{source}_success'] = r.get('training_success', False)
            row[f'from_{source}_skipped'] = r.get('skipped', False)
            row[f'best_from_{source}_path'] = r.get('best_from_checkpoint_path')
            row[f'last_from_{source}_path'] = r.get('last_from_checkpoint_path')
        rows.append(row)

    csv_file = results_root / f"finetune_summary_run{run_idx}_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    print(f"✅ CSV summary saved to: {csv_file}")


# ============================================================================
# COMMAND LINE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune KenshoDense-pretrained backbones (DeepLabV3) on F3.")
    parser.add_argument("--runs", type=int, nargs="+", default=[1],
                        help="Pretrain run indices to finetune (default: 1).")
    parser.add_argument("--width", type=int, choices=(1, 2, 4), default=1,
                        help="Backbone width used in pretraining (selects the "
                             "checkpoint folders; default: 1).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=int, default=0, help="GPU index (default: 0).")
    parser.add_argument("--from", dest="sources", choices=("last", "best", "both"), default="last",
                        help="Which pretrain checkpoint to finetune from (default: last).")
    parser.add_argument("--families", nargs="+", choices=FAMILIES)
    parser.add_argument("--configs", nargs="+", help="e.g. GB_param1 FF_param3")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="Limit the number of configurations (for testing).")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-finetune checkpoints that were already processed.")
    parser.add_argument("--clean-logs", action="store_true",
                        help="Delete old finetuning logs before starting.")
    parser.add_argument("--data-root", type=Path, default=paths.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=paths.DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sources = ("best", "last") if args.sources == "both" else (args.sources,)

    print("=" * 80)
    print(f"{MODEL_NAME} — batch finetuning")
    print(f"Backbone: ResNet-50 ({args.width}x width) | runs: {args.runs}")
    print("=" * 80)

    if args.clean_logs:
        logs_root = paths.logs_dir(args.output_root, "finetune")
        if logs_root.exists():
            print(f"Cleaning old logs from {logs_root}...")
            shutil.rmtree(logs_root, ignore_errors=True)

    for run_idx in args.runs:
        finetune_run(
            run_idx=run_idx,
            width_multiplier=args.width,
            data_root=args.data_root,
            output_root=args.output_root,
            sources=sources,
            finetune_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            max_experiments=args.max_experiments,
            skip_completed=not args.no_skip,
            families=args.families,
            configs=args.configs,
        )

    print("\n" + "=" * 80)
    print(f"✨ Done: run index(es) {args.runs}")
    print("=" * 80)
