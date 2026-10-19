"""
evaluate.py
Test-set evaluation of the finetuned DeepLabV3 checkpoints produced by
finetune.py: mIoU, accuracy, precision, recall, F1 (macro and per class)
and the confusion matrix.

For each run index, every checkpoint matching ``--pattern`` in
``<output_root>/checkpoints/finetune/rn50x{k}/run{r}/`` is evaluated and its
results are written to ``<output_root>/evaluation/rn50x{k}/run{r}/<ckpt>/``.
A combined table of all evaluated checkpoints (one row per configuration
and run, ready for per-configuration statistics and paired tests) is
written to ``<output_root>/evaluation/rn50x{k}/all_runs_summary.csv``.

Examples
--------
    python evaluate.py --runs 1 2 3 4 5 6 --width 4
    python evaluate.py --checkpoint path/to/GB_param1_run1_best_from_last.ckpt
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import pandas as pd
import torch
from torchmetrics import JaccardIndex, Accuracy, Precision, Recall, F1Score
from torchmetrics.classification import MulticlassConfusionMatrix

from minerva.data.data_modules.f3 import F3DataModule, default_test_transforms

from deeplabv3 import DeepLabV3, DeepLabV3Backbone
import paths

warnings.filterwarnings('ignore')

# Finetuned checkpoint stems produced by finetune.py:
#   <config>_run<r>_<best|last>_from_<best|last>
FINETUNED_PATTERN = re.compile(
    r'^(?P<config>.+)_run(?P<run>\d+)_(?P<ft>best|last)_from_(?P<src>best|last)$'
)


class Evaluator:
    """Metrics-only evaluation of finetuned segmentation models."""

    def __init__(self, num_classes: int = paths.NUM_CLASSES):
        self.num_classes = num_classes

        self.metrics = {
            'jaccard': JaccardIndex(task="multiclass", num_classes=num_classes, average='macro'),
            'accuracy': Accuracy(task="multiclass", num_classes=num_classes, average='macro'),
            'precision': Precision(task="multiclass", num_classes=num_classes, average='macro'),
            'recall': Recall(task="multiclass", num_classes=num_classes, average='macro'),
            'f1': F1Score(task="multiclass", num_classes=num_classes, average='macro'),
            'confusion_matrix': MulticlassConfusionMatrix(num_classes=num_classes),
        }

        self.per_class_metrics = {
            'jaccard_per_class': JaccardIndex(task="multiclass", num_classes=num_classes, average=None),
            'accuracy_per_class': Accuracy(task="multiclass", num_classes=num_classes, average=None),
            'precision_per_class': Precision(task="multiclass", num_classes=num_classes, average=None),
            'recall_per_class': Recall(task="multiclass", num_classes=num_classes, average=None),
            'f1_per_class': F1Score(task="multiclass", num_classes=num_classes, average=None),
        }

    def load_model_from_checkpoint(self, checkpoint_path: str) -> DeepLabV3:
        """Load a finetuned model, rebuilding the backbone with the width
        (RN50 1x/2x/4x) the checkpoint was trained with."""
        print(f"Loading model from checkpoint: {checkpoint_path}")

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        hparams = checkpoint.get("hyper_parameters", {})

        # Infer width from the stem conv: 64 * k output channels.
        stem_key = "backbone.RN50model.conv1.weight"
        if stem_key not in state_dict:
            raise RuntimeError(f"'{stem_key}' not found in checkpoint; cannot infer backbone width. "
                               f"Example keys: {list(state_dict.keys())[:5]}")
        width_multiplier = state_dict[stem_key].shape[0] // 64
        print(f"Detected backbone width: RN50 {width_multiplier}x")

        backbone = DeepLabV3Backbone(pretrained=False, width_multiplier=width_multiplier)
        model = DeepLabV3(
            backbone=backbone,
            num_classes=self.num_classes,
            learning_rate=hparams.get("learning_rate", 0.001),  # irrelevant for eval
            pretrained=False,
        )

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        print("Model loaded successfully")
        return model

    def prepare_data_module(self, data_root: Path, batch_size: int = 16) -> F3DataModule:
        """Prepare the F3 test split."""
        data_module = F3DataModule(
            root_data_dir=str(paths.images_dir(data_root)),
            root_annotation_dir=str(paths.annotations_dir(data_root)),
            batch_size=batch_size,
            num_workers=4,
            drop_last=False,
            test_transforms=default_test_transforms(img_size=paths.IMG_SIZE),
        )
        data_module.setup("test")
        return data_module

    def evaluate_model(self, model: DeepLabV3, data_module: F3DataModule) -> dict:
        """Run the model over the whole test split and compute all metrics."""
        print("Starting evaluation...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_module.test_dataloader()):
                images, targets = batch[0], batch[1]

                images = images.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.squeeze(1).cpu() if targets.dim() > 3 else targets.cpu())

                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches...")

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        print(f"Evaluation completed on {len(predictions)} samples")

        results = self._calculate_metrics(predictions, targets)
        results['num_samples'] = len(predictions)
        return results

    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute macro, per-class and distribution metrics over all pixels."""
        results = {}

        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()

        for name, metric in self.metrics.items():
            value = metric(predictions_flat, targets_flat)
            results[name] = value if name == 'confusion_matrix' else value.item()

        results['per_class_metrics'] = {
            name.replace('_per_class', ''): metric(predictions_flat, targets_flat).tolist()
            for name, metric in self.per_class_metrics.items()
        }

        predictions_clean = predictions_flat.clamp(min=0).long()
        targets_clean = targets_flat.clamp(min=0).long()
        results['class_distribution_pred'] = torch.bincount(predictions_clean, minlength=self.num_classes).tolist()
        results['class_distribution_true'] = torch.bincount(targets_clean, minlength=self.num_classes).tolist()

        return results

    def save_evaluation_results(self, results: dict, save_dir: Path, checkpoint_name: str) -> Path:
        """Save metrics.json, summary.csv and per_class_metrics.csv."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving results to: {save_dir}")

        metrics_json = {
            'checkpoint_name': checkpoint_name,
            'num_samples': results['num_samples'],
            'overall_metrics': {
                'jaccard': float(results['jaccard']),
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1': float(results['f1']),
            },
            'per_class_metrics': results['per_class_metrics'],
            'class_distribution': {
                'true': results['class_distribution_true'],
                'predicted': results['class_distribution_pred'],
            },
            'confusion_matrix': results['confusion_matrix'].tolist(),
        }

        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_json, f, indent=2)

        pd.DataFrame({
            'Metric': ['mIoU', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [results['jaccard'], results['accuracy'], results['precision'],
                      results['recall'], results['f1']],
        }).to_csv(save_dir / 'summary.csv', index=False)

        per_class_df = pd.DataFrame(results['per_class_metrics'])
        per_class_df.insert(0, 'Class', [f'Class_{i}' for i in range(len(per_class_df))])
        per_class_df.to_csv(save_dir / 'per_class_metrics.csv', index=False)

        print("Results saved successfully")
        return save_dir


def evaluate_single_checkpoint(checkpoint_path: str, output_dir: Path, data_root: Path,
                               batch_size: int = 16) -> dict:
    """Evaluate a single checkpoint and save its results."""
    output_dir = Path(output_dir)

    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")

    evaluator = Evaluator()
    model = evaluator.load_model_from_checkpoint(checkpoint_path)
    data_module = evaluator.prepare_data_module(data_root, batch_size=batch_size)
    results = evaluator.evaluate_model(model, data_module)
    evaluator.save_evaluation_results(results, output_dir, Path(checkpoint_path).stem)

    print(f"mIoU: {results['jaccard']*100:.2f}%")
    return results


def evaluate_directory(checkpoint_dir: Path, output_dir: Path, data_root: Path,
                       pattern: str = "*best_from_last.ckpt", batch_size: int = 16) -> dict:
    """Evaluate every checkpoint matching ``pattern`` in ``checkpoint_dir``."""
    checkpoint_dir = Path(checkpoint_dir)
    print(f"Searching for checkpoints in: {checkpoint_dir}")

    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return {}

    all_checkpoints = sorted(checkpoint_dir.glob(pattern))
    print(f"Pattern '{pattern}': {len(all_checkpoints)} files")
    if not all_checkpoints:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_summary = {}

    for i, ckpt_path in enumerate(all_checkpoints, 1):
        ckpt_name = ckpt_path.stem
        print(f"\n{'='*80}")
        print(f"Checkpoint {i}/{len(all_checkpoints)}: {ckpt_name}")
        print(f"{'='*80}")

        match = FINETUNED_PATTERN.match(ckpt_name)
        name_info = {
            'config_key': match.group('config') if match else None,
            'run': int(match.group('run')) if match else None,
        }

        try:
            ckpt_output = output_dir / ckpt_name
            results = evaluate_single_checkpoint(str(ckpt_path), ckpt_output, data_root, batch_size)
            results_summary[ckpt_name] = {
                'status': 'SUCCESS',
                **name_info,
                'checkpoint_path': str(ckpt_path),
                'miou': results['jaccard'],
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1'],
                'num_samples': results['num_samples'],
                'results_dir': str(ckpt_output),
            }
            print("✅ Evaluation completed successfully!")
        except Exception as e:
            print(f"❌ Error evaluating {ckpt_name}: {str(e)}")
            results_summary[ckpt_name] = {
                'status': 'ERROR',
                **name_info,
                'error': str(e),
                'checkpoint_path': str(ckpt_path),
            }

    with open(output_dir / "batch_evaluation_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    rows = [{'checkpoint': name, **{k: v for k, v in r.items()
                                    if k in ('config_key', 'run', 'miou', 'accuracy',
                                             'precision', 'recall', 'f1_score')}}
            for name, r in results_summary.items() if r['status'] == 'SUCCESS']
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "batch_summary.csv", index=False)

    successful = sum(1 for v in results_summary.values() if v['status'] == 'SUCCESS')
    print(f"\nSuccessfully evaluated: {successful}/{len(results_summary)} -> {output_dir}")
    return results_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate finetuned DeepLabV3 checkpoints on the F3 test split.")
    parser.add_argument("--runs", type=int, nargs="+", default=[1])
    parser.add_argument("--width", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--pattern", default="*best_from_last.ckpt",
                        help="Glob of finetuned checkpoints to evaluate (default: *best_from_last.ckpt).")
    parser.add_argument("--checkpoint", type=Path,
                        help="Evaluate a single checkpoint instead of whole run folders.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=paths.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=paths.DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.checkpoint is not None:
        out = Path(args.output_root) / "evaluation" / "single" / args.checkpoint.stem
        evaluate_single_checkpoint(str(args.checkpoint), out, args.data_root, args.batch_size)
        raise SystemExit(0)

    combined = []
    for run_idx in args.runs:
        summary = evaluate_directory(
            checkpoint_dir=paths.finetune_ckpt_dir(args.output_root, args.width, run_idx),
            output_dir=paths.evaluation_dir(args.output_root, args.width, run_idx),
            data_root=args.data_root,
            pattern=args.pattern,
            batch_size=args.batch_size,
        )
        combined.extend(
            {'checkpoint': name, **r} for name, r in summary.items() if r['status'] == 'SUCCESS'
        )

    if combined:
        width_root = Path(args.output_root) / "evaluation" / paths.width_tag(args.width)
        df = pd.DataFrame(combined)[['config_key', 'run', 'checkpoint', 'miou', 'accuracy',
                                     'precision', 'recall', 'f1_score']]
        df.sort_values(['config_key', 'run']).to_csv(width_root / "all_runs_summary.csv", index=False)
        print(f"\nCombined summary: {width_root / 'all_runs_summary.csv'}")
