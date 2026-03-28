#!/usr/bin/env python3
"""
BERT fine-tuning for 3-class misconception detection.

3-class mapping:
    True_Correct        -> 0
    False_Misconception -> 1
    True_Misconception  -> 2

Supports:
    - Standard cross-entropy loss
    - Weighted cross-entropy loss (inverse class frequency)
    - Focal loss (Lin et al. 2017) with per-class alpha weights
    - Question-level rebalancing within TM class (per-sample weighting)

Usage:
    python scripts/train_bert.py --loss_type standard --output_dir models/bert-test/
    python scripts/train_bert.py --loss_type weighted --output_dir models/bert-weighted/
    python scripts/train_bert.py --loss_type weighted --question_weighted --output_dir models/bert-qw/
    python scripts/train_bert.py --loss_type focal --focal_gamma 2.0 --output_dir models/bert-focal/
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

LABEL2ID = {
    'True_Correct': 0,
    'False_Misconception': 1,
    'True_Misconception': 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)


class MisconceptionDataset(Dataset):
    """
    Each JSONL item has:
        - input: str (question + student answer + explanation)
        - output: str (diagnosis -- not used for classification)
        - meta: dict with category, question_id, etc.
    """

    def __init__(self, filepath: Path, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        category = item['meta']['category']
        label = LABEL2ID[category]

        encoding = self.tokenizer(
            item['input'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'meta': item['meta'],
        }


def collate_fn(batch: list) -> dict:
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'label': torch.stack([x['label'] for x in batch]),
        'meta': [x['meta'] for x in batch],
    }


def compute_class_weights(dataset: MisconceptionDataset) -> torch.Tensor:
    """Inverse-frequency class weights: weight_c = N / (num_classes * count_c)."""
    counts = Counter()
    for item in dataset.data:
        counts[item['meta']['category']] += 1

    n = len(dataset)
    weights = torch.zeros(NUM_CLASSES)
    for label_name, label_id in LABEL2ID.items():
        count = counts.get(label_name, 1)
        weights[label_id] = n / (NUM_CLASSES * count)

    return weights


def compute_question_weights(dataset: MisconceptionDataset) -> dict:
    """
    Per-question inverse-frequency weights for TM samples.
    Normalised so the mean weight across all TM samples is 1.0.
    """
    tm_counts = Counter()
    for item in dataset.data:
        if item['meta']['category'] == 'True_Misconception':
            tm_counts[item['meta']['question_id']] += 1

    if not tm_counts:
        return {}

    raw = {qid: 1.0 / count for qid, count in tm_counts.items()}
    total_tm = sum(tm_counts.values())
    total_raw_weight = sum(raw[qid] * tm_counts[qid] for qid in tm_counts)
    normalisation = total_tm / total_raw_weight

    return {qid: raw[qid] * normalisation for qid in raw}


class FocalLoss(nn.Module):
    """
    Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 down-weights easy examples, focusing on hard boundary cases.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, reduction='none',
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimiser: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    question_weights: Optional[dict] = None,
    focal_loss_fn: Optional[FocalLoss] = None,
) -> float:
    """
    Loss selection priority:
        1. focal_loss_fn -- FocalLoss module
        2. question_weights -- per-sample weighting with unreduced CE
        3. class_weights -- standard weighted CrossEntropyLoss
        4. None -- unweighted CrossEntropyLoss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc='  Training', leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        batch_size = labels.size(0)

        optimiser.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if focal_loss_fn is not None:
            loss = focal_loss_fn(logits, labels)
        elif question_weights is not None:
            unreduced_loss = nn.functional.cross_entropy(
                logits, labels, reduction='none'
            )
            batch_weights = torch.zeros(batch_size, device=device)
            for j, meta in enumerate(batch['meta']):
                label_id = LABEL2ID[meta['category']]
                w = class_weights[label_id].item() if class_weights is not None else 1.0
                if meta['category'] == 'True_Misconception':
                    qid = meta['question_id']
                    w *= question_weights.get(qid, 1.0)
                batch_weights[j] = w
            loss = (unreduced_loss * batch_weights).mean()
        else:
            loss = nn.functional.cross_entropy(
                logits, labels,
                weight=class_weights.to(device) if class_weights is not None else None,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_weights_tensor: Optional[torch.Tensor] = None,
) -> tuple[float, list[dict]]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    predictions = []

    for batch in tqdm(dataloader, desc='  Evaluating', leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Unweighted loss for fair comparison across configs
        loss = nn.functional.cross_entropy(logits, labels)
        total_loss += loss.item()
        num_batches += 1

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

        for j, meta in enumerate(batch['meta']):
            pred_label = ID2LABEL[int(preds[j])]
            true_label = meta['category']
            prob_dict = {ID2LABEL[k]: float(probs[j][k]) for k in range(NUM_CLASSES)}

            predictions.append({
                'question_id': meta['question_id'],
                'row_id': meta.get('row_id'),
                'misconception_code': meta.get('misconception_code'),
                'true_label': true_label,
                'pred_label': pred_label,
                'pred_proba': prob_dict,
            })

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, predictions


def compute_per_class_metrics(predictions: list[dict]) -> dict:
    counts = {}
    for cls in LABEL2ID:
        tp = sum(1 for p in predictions if p['true_label'] == cls and p['pred_label'] == cls)
        fp = sum(1 for p in predictions if p['true_label'] != cls and p['pred_label'] == cls)
        fn = sum(1 for p in predictions if p['true_label'] == cls and p['pred_label'] != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        counts[cls] = {'precision': prec, 'recall': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

    accuracy = sum(1 for p in predictions if p['true_label'] == p['pred_label']) / max(len(predictions), 1)
    return {'per_class': counts, 'accuracy': accuracy}


def print_metrics(metrics: dict, prefix: str = '') -> None:
    print(f"{prefix}Overall accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix}{'Class':<25s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}")
    print(f"{prefix}{'-' * 49}")
    for cls in LABEL2ID:
        m = metrics['per_class'][cls]
        short = {'True_Correct': 'TC', 'False_Misconception': 'FM', 'True_Misconception': 'TM'}[cls]
        print(f"{prefix}{short:<25s} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")


def get_data_files(input_variant: str) -> tuple[Path, Path, Path]:
    """Return (train, val, test) paths for a given input variant."""
    if input_variant == 'original':
        base = ROOT / 'data' / 'processed'
        return base / 'train.jsonl', base / 'val.jsonl', base / 'test.jsonl'
    else:
        base = ROOT / 'data' / 'variants'
        suffix = f'_{input_variant}'
        return (
            base / f'train{suffix}.jsonl',
            base / f'val{suffix}.jsonl',
            base / f'test{suffix}.jsonl',
        )


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune BERT for 3-class misconception detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--loss_type', choices=['standard', 'weighted', 'focal'],
                        default='standard')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss focusing parameter')
    parser.add_argument('--class_weights', type=float, nargs=3, default=None,
                        metavar=('TC', 'FM', 'TM'),
                        help='Custom class weights. If unset with weighted loss, uses inverse frequency.')
    parser.add_argument('--question_weighted', action='store_true',
                        help='Per-question inverse-frequency weighting within TM class')
    parser.add_argument('--input_variant', choices=['original', 'masked', 'swapped'],
                        default='original')
    parser.add_argument('--train_file', type=str, default=None,
                        help='Custom training JSONL (overrides --input_variant)')
    parser.add_argument('--val_file', type=str, default=None,
                        help='Custom validation JSONL (overrides --input_variant)')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Custom test JSONL (overrides --input_variant)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save model checkpoint and results')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)

    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--sanity_check', action='store_true',
                        help='Quick test on 100 samples for 2 epochs')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print('=' * 70)
    print('BERT Training Pipeline')
    print('=' * 70)
    print(f'Device:            {device}')
    print(f'Model:             {args.model_name}')
    print(f'Loss type:         {args.loss_type}')
    if args.loss_type == 'focal':
        print(f'Focal gamma:       {args.focal_gamma}')
    print(f'Question weighted: {args.question_weighted}')
    print(f'Input variant:     {args.input_variant}')
    print(f'Output dir:        {args.output_dir}')
    print(f'Epochs:            {args.epochs}')
    print(f'Batch size:        {args.batch_size}')
    print(f'Learning rate:     {args.learning_rate}')
    print(f'Max length:        {args.max_length}')
    print(f'Patience:          {args.patience}')
    print(f'Seed:              {args.seed}')
    print()

    print('Label mapping:')
    for name, idx in LABEL2ID.items():
        print(f'  {name} -> {idx}')
    print()

    # Data loading -- allow overriding individual files
    default_train, default_val, default_test = get_data_files(args.input_variant)
    train_path = Path(args.train_file) if args.train_file else default_train
    val_path = Path(args.val_file) if args.val_file else default_val
    test_path = Path(args.test_file) if args.test_file else default_test
    print(f'Data files:')
    print(f'  Train: {train_path}')
    print(f'  Val:   {val_path}')
    print(f'  Test:  {test_path}')
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = MisconceptionDataset(train_path, tokenizer, args.max_length)
    val_dataset = MisconceptionDataset(val_path, tokenizer, args.max_length)
    test_dataset = MisconceptionDataset(test_path, tokenizer, args.max_length)

    if args.sanity_check:
        print('SANITY CHECK: 100 train, 50 val, 50 test, 2 epochs')
        train_dataset.data = train_dataset.data[:100]
        val_dataset.data = val_dataset.data[:50]
        test_dataset.data = test_dataset.data[:50]
        args.epochs = 2
        args.patience = 10
        print()

    for name, ds in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        cats = Counter(item['meta']['category'] for item in ds.data)
        print(f'{name}: {len(ds)} samples')
        for cat in LABEL2ID:
            count = cats.get(cat, 0)
            pct = 100 * count / max(len(ds), 1)
            print(f'  {cat}: {count} ({pct:.1f}%)')
    print()

    # Loss setup
    class_weights_tensor = None
    question_weights_dict = None
    focal_loss_fn = None

    if args.loss_type in ('weighted', 'focal'):
        if args.class_weights is not None:
            class_weights_tensor = torch.tensor(args.class_weights, dtype=torch.float32)
            print(f'Using custom class weights: {args.class_weights}')
        else:
            class_weights_tensor = compute_class_weights(train_dataset)
            print('Using inverse-frequency class weights:')

        for name, idx in LABEL2ID.items():
            print(f'  {name} (id={idx}): {class_weights_tensor[idx]:.4f}')

        tm_w = class_weights_tensor[LABEL2ID['True_Misconception']].item()
        tc_w = class_weights_tensor[LABEL2ID['True_Correct']].item()
        print(f'  TM/TC weight ratio: {tm_w / tc_w:.1f}x')
        print()

    if args.loss_type == 'focal':
        focal_loss_fn = FocalLoss(alpha=class_weights_tensor, gamma=args.focal_gamma)
        print(f'Focal loss: gamma={args.focal_gamma}, alpha=class_weights')
        print()

    if args.question_weighted:
        question_weights_dict = compute_question_weights(train_dataset)
        print('Question-level weights for TM class (normalised, mean=1.0):')
        for qid in sorted(question_weights_dict.keys()):
            tm_count = sum(
                1 for item in train_dataset.data
                if item['meta']['category'] == 'True_Misconception'
                and item['meta']['question_id'] == qid
            )
            print(f'  Q{qid}: weight={question_weights_dict[qid]:.4f} (TM train count: {tm_count})')
        print()

    # Model
    print(f'Loading model: {args.model_name}')
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=NUM_CLASSES, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model.to(device)
    if focal_loss_fn is not None:
        focal_loss_fn.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters:     {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f'Total training steps:  {total_steps}')
    print(f'Warmup steps:          {warmup_steps}')
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    training_log = {
        'config': {
            'model_name': args.model_name,
            'loss_type': args.loss_type,
            'focal_gamma': args.focal_gamma if args.loss_type == 'focal' else None,
            'class_weights': class_weights_tensor.tolist() if class_weights_tensor is not None else None,
            'question_weighted': args.question_weighted,
            'question_weights': {str(k): v for k, v in question_weights_dict.items()} if question_weights_dict else None,
            'train_file': str(train_path),
            'input_variant': args.input_variant,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_length': args.max_length,
            'patience': args.patience,
            'seed': args.seed,
            'warmup_ratio': args.warmup_ratio,
        },
        'label_mapping': LABEL2ID,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'epochs': [],
        'best_epoch': None,
        'best_val_loss': float('inf'),
        'started_at': datetime.now().isoformat(),
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    print('=' * 70)
    print('Starting training')
    print('=' * 70)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        print('-' * 40)

        train_loss = train_one_epoch(
            model, train_loader, optimiser, scheduler, device,
            class_weights=class_weights_tensor,
            question_weights=question_weights_dict,
            focal_loss_fn=focal_loss_fn,
        )

        val_loss, val_predictions = evaluate_epoch(model, val_loader, device)
        val_metrics = compute_per_class_metrics(val_predictions)
        epoch_time = time.time() - epoch_start

        print(f'\n  Train loss: {train_loss:.4f}')
        print(f'  Val loss:   {val_loss:.4f}')
        print(f'  Epoch time: {epoch_time:.1f}s')
        print()
        print('  Validation metrics:')
        print_metrics(val_metrics, prefix='    ')

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'epoch_time_s': round(epoch_time, 1),
        }
        training_log['epochs'].append(epoch_log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            training_log['best_epoch'] = epoch + 1
            training_log['best_val_loss'] = val_loss
            print(f'\n  New best val loss! Saving checkpoint to {output_dir}')
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            epochs_without_improvement += 1
            print(f'\n  No improvement for {epochs_without_improvement}/{args.patience} epochs')
            if epochs_without_improvement >= args.patience:
                print(f'\n  Early stopping after {epoch + 1} epochs.')
                break

    print()
    print('=' * 70)
    print('Training complete')
    print(f'Best val loss: {best_val_loss:.4f} (epoch {training_log["best_epoch"]})')
    print('=' * 70)

    # Test evaluation with best checkpoint
    print(f'\nLoading best checkpoint from {output_dir}...')
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)

    test_loss, test_predictions = evaluate_epoch(model, test_loader, device)
    test_metrics = compute_per_class_metrics(test_predictions)

    print(f'\nTest loss: {test_loss:.4f}')
    print('\nTest metrics:')
    print_metrics(test_metrics, prefix='  ')

    training_log['test_loss'] = test_loss
    training_log['test_metrics'] = test_metrics

    # Save predictions
    predictions_path = output_dir / 'test_predictions.jsonl'
    print(f'\nSaving {len(test_predictions)} test predictions to {predictions_path}')
    with open(predictions_path, 'w') as f:
        for pred in test_predictions:
            f.write(json.dumps(pred) + '\n')

    # Run evaluation harness if available
    harness_results = None
    try:
        sys.path.insert(0, str(ROOT / 'scripts'))
        from evaluation_harness import evaluate as harness_evaluate

        print('\nRunning evaluation harness on test predictions...')
        harness_result = harness_evaluate(test_predictions)

        harness_json_path = output_dir / 'evaluation_results.json'
        with open(harness_json_path, 'w') as f:
            f.write(harness_result.to_json())
        print(f'Evaluation JSON saved to {harness_json_path}')

        harness_md_path = output_dir / 'evaluation_results.md'
        with open(harness_md_path, 'w') as f:
            f.write(harness_result.to_markdown())
        print(f'Evaluation markdown saved to {harness_md_path}')

        print('\n' + harness_result.to_markdown())
        harness_results = harness_result.to_dict()
    except ImportError:
        print('\nWarning: Could not import evaluation_harness. Run manually:')
        print(f'  python scripts/evaluation_harness.py --predictions {predictions_path}')
    except Exception as e:
        print(f'\nWarning: Evaluation harness failed: {e}')

    # Save training log
    training_log['finished_at'] = datetime.now().isoformat()
    if harness_results is not None:
        training_log['harness_results'] = harness_results

    log_path = output_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f'\nTraining log saved to {log_path}')

    print('\n' + '=' * 70)
    print('Done.')
    print('=' * 70)

    return training_log


if __name__ == '__main__':
    main()
