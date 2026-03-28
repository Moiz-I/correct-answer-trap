#!/usr/bin/env python3
"""
Shared evaluation harness for misconception detection models.

Computes micro-averaged, macro-averaged (across questions), per-question,
confusion matrix, and bootstrap CI metrics for the 3-class problem:
True_Correct (TC), False_Misconception (FM), True_Misconception (TM).

Usage as module:
    from scripts.evaluation_harness import evaluate, evaluate_from_file

Usage as CLI:
    python scripts/evaluation_harness.py --predictions preds.jsonl \
        --output-json results.json --output-md results.md
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path


CLASS_ORDER = ['True_Correct', 'False_Misconception', 'True_Misconception']
CLASS_SHORT = {'True_Correct': 'TC', 'False_Misconception': 'FM', 'True_Misconception': 'TM'}

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)
    return {'precision': precision, 'recall': recall, 'f1': f1}


def _confusion_counts(true_labels: list[str], pred_labels: list[str]) -> dict:
    counts = {}
    for cls in CLASS_ORDER:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        counts[cls] = {'tp': tp, 'fp': fp, 'fn': fn}
    return counts


def _confusion_matrix(true_labels: list[str], pred_labels: list[str]) -> list[list[int]]:
    """3x3 confusion matrix. Rows = true, columns = predicted."""
    matrix = [[0] * len(CLASS_ORDER) for _ in CLASS_ORDER]
    idx = {cls: i for i, cls in enumerate(CLASS_ORDER)}
    for t, p in zip(true_labels, pred_labels):
        if t in idx and p in idx:
            matrix[idx[t]][idx[p]] += 1
    return matrix


def compute_micro_metrics(true_labels: list[str], pred_labels: list[str]) -> dict:
    counts = _confusion_counts(true_labels, pred_labels)
    per_class = {}
    for cls in CLASS_ORDER:
        c = counts[cls]
        per_class[cls] = _precision_recall_f1(c['tp'], c['fp'], c['fn'])

    accuracy = (sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
                / len(true_labels) if true_labels else 0.0)

    return {'per_class': per_class, 'accuracy': accuracy}


def compute_macro_metrics(predictions: list[dict]) -> dict:
    """
    Per-class P/R/F1 averaged across questions.

    Questions with zero true instances of a class are excluded from
    that class's recall/F1 average. Questions with zero predicted
    instances are excluded from precision average.
    """
    by_question: dict[int, list[dict]] = {}
    for pred in predictions:
        qid = pred['question_id']
        by_question.setdefault(qid, []).append(pred)

    per_question_metrics: dict[int, dict] = {}
    for qid, preds in by_question.items():
        true_labels = [p['true_label'] for p in preds]
        pred_labels = [p['pred_label'] for p in preds]
        counts = _confusion_counts(true_labels, pred_labels)
        q_metrics = {}
        for cls in CLASS_ORDER:
            c = counts[cls]
            q_metrics[cls] = {
                **_precision_recall_f1(c['tp'], c['fp'], c['fn']),
                'n_true': c['tp'] + c['fn'],
                'n_pred': c['tp'] + c['fp'],
            }
        per_question_metrics[qid] = q_metrics

    macro = {}
    for cls in CLASS_ORDER:
        precisions = []
        recalls = []
        f1s = []
        for qid, q_metrics in per_question_metrics.items():
            m = q_metrics[cls]
            if m['n_true'] > 0:
                recalls.append(m['recall'])
                f1s.append(m['f1'])
            if m['n_pred'] > 0:
                precisions.append(m['precision'])

        macro[cls] = {
            'precision': _safe_mean(precisions),
            'recall': _safe_mean(recalls),
            'f1': _safe_mean(f1s),
            'n_questions_with_true': sum(
                1 for qid in per_question_metrics
                if per_question_metrics[qid][cls]['n_true'] > 0
            ),
        }

    return macro


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_per_question_breakdown(predictions: list[dict]) -> list[dict]:
    by_question: dict[int, list[dict]] = {}
    for pred in predictions:
        qid = pred['question_id']
        by_question.setdefault(qid, []).append(pred)

    rows = []
    for qid in sorted(by_question.keys()):
        preds = by_question[qid]
        true_labels = [p['true_label'] for p in preds]
        pred_labels = [p['pred_label'] for p in preds]

        n_tc = sum(1 for t in true_labels if t == 'True_Correct')
        n_fm = sum(1 for t in true_labels if t == 'False_Misconception')
        n_tm = sum(1 for t in true_labels if t == 'True_Misconception')

        counts = _confusion_counts(true_labels, pred_labels)

        row = {
            'question_id': qid,
            'n_samples': len(preds),
            'n_tc': n_tc,
            'n_fm': n_fm,
            'n_tm': n_tm,
        }
        for cls in CLASS_ORDER:
            short = CLASS_SHORT[cls]
            c = counts[cls]
            n_true = c['tp'] + c['fn']
            row[f'recall_{short}'] = c['tp'] / n_true if n_true > 0 else None
        rows.append(row)

    return rows


def _bootstrap_metric(
    predictions: list[dict],
    metric_fn,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap 95% CI for a metric function that takes predictions and returns a float."""
    rng = random.Random(seed)
    n = len(predictions)
    values = []
    for _ in range(n_bootstrap):
        sample = [predictions[rng.randint(0, n - 1)] for _ in range(n)]
        values.append(metric_fn(sample))
    values.sort()
    lo = int(n_bootstrap * 0.025)
    hi = int(n_bootstrap * 0.975) - 1
    return {
        'mean': _safe_mean(values),
        'ci_lower': values[lo],
        'ci_upper': values[max(hi, 0)],
    }


def _micro_tm_recall(predictions: list[dict]) -> float:
    tp = sum(1 for p in predictions
             if p['true_label'] == 'True_Misconception'
             and p['pred_label'] == 'True_Misconception')
    fn = sum(1 for p in predictions
             if p['true_label'] == 'True_Misconception'
             and p['pred_label'] != 'True_Misconception')
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _macro_tm_recall(predictions: list[dict]) -> float:
    macro = compute_macro_metrics(predictions)
    return macro['True_Misconception']['recall']


def compute_bootstrap_cis(predictions: list[dict],
                          n_bootstrap: int = BOOTSTRAP_N,
                          seed: int = BOOTSTRAP_SEED) -> dict:
    return {
        'micro_tm_recall': _bootstrap_metric(
            predictions, _micro_tm_recall, n_bootstrap, seed),
        'macro_tm_recall': _bootstrap_metric(
            predictions, _macro_tm_recall, n_bootstrap, seed),
    }


@dataclass
class EvaluationResult:
    micro_metrics: dict = field(default_factory=dict)
    macro_metrics: dict = field(default_factory=dict)
    per_question: list = field(default_factory=list)
    confusion_matrix: list = field(default_factory=list)
    bootstrap_cis: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'micro_metrics': self.micro_metrics,
            'macro_metrics': self.macro_metrics,
            'per_question': self.per_question,
            'confusion_matrix': {
                'labels': CLASS_ORDER,
                'matrix': self.confusion_matrix,
            },
            'bootstrap_cis': self.bootstrap_cis,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

def evaluate(predictions: list[dict],
             n_bootstrap: int = BOOTSTRAP_N,
             bootstrap_seed: int = BOOTSTRAP_SEED) -> EvaluationResult:
    """
    Run full evaluation on a list of prediction dicts.

    Each dict must have: question_id (int), true_label (str), pred_label (str).
    Optional: pred_proba (dict mapping class names to probabilities).
    """
    true_labels = [p['true_label'] for p in predictions]
    pred_labels = [p['pred_label'] for p in predictions]

    return EvaluationResult(
        micro_metrics=compute_micro_metrics(true_labels, pred_labels),
        macro_metrics=compute_macro_metrics(predictions),
        per_question=compute_per_question_breakdown(predictions),
        confusion_matrix=_confusion_matrix(true_labels, pred_labels),
        bootstrap_cis=compute_bootstrap_cis(predictions, n_bootstrap, bootstrap_seed),
    )


def evaluate_from_file(filepath: str | Path,
                       n_bootstrap: int = BOOTSTRAP_N,
                       bootstrap_seed: int = BOOTSTRAP_SEED) -> EvaluationResult:
    path = Path(filepath)
    predictions = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return evaluate(predictions, n_bootstrap, bootstrap_seed)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate misconception detection predictions.')
    parser.add_argument('--predictions', required=True,
                        help='Path to predictions JSONL file')
    parser.add_argument('--output-json', default=None,
                        help='Path to write JSON results')
                        help='Path to write Markdown results')
    parser.add_argument('--bootstrap-n', type=int, default=BOOTSTRAP_N)
    parser.add_argument('--bootstrap-seed', type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    result = evaluate_from_file(
        args.predictions,
        n_bootstrap=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
    )

    print(result.to_json())

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w') as f:
            f.write(result.to_json())
        print(f'\nJSON results written to {args.output_json}')



if __name__ == '__main__':
    main()
