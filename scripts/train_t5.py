#!/usr/bin/env python3
"""T5-small fine-tuning with LoRA for misconception detection."""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "models" / "t5-small-lora"
RESULTS_DIR = ROOT / "data" / "finetuned_results"


@dataclass
class TrainingConfig:
    model_name: str = "t5-small"
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 2e-4
    max_input_length: int = 512
    max_output_length: int = 128
    warmup_ratio: float = 0.1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: tuple = ("q", "v")


class MisconceptionDataset(Dataset):
    def __init__(self, filepath: Path, tokenizer: T5Tokenizer, config: TrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.data = []

        with open(filepath, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        inputs = self.tokenizer(
            item["input"],
            max_length=self.config.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        outputs = self.tokenizer(
            item["output"],
            max_length=self.config.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token id with -100 so it is ignored in loss
        labels = outputs["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
            "meta": item["meta"],
        }


def collate_fn(batch: list) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "meta": [x["meta"] for x in batch],
    }


def parse_binary_prediction(text: str) -> str:
    text_lower = text.lower().strip()
    if text_lower == "no_misconception" or "no misconception" in text_lower:
        return "no_misconception"
    return "misconception"


def evaluate(
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss, {}


def generate_predictions(
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    config: TrainingConfig,
    device: torch.device,
    original_data: list,
) -> list[dict]:
    model.eval()
    predictions = []
    data_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.max_output_length,
                num_beams=1,
                do_sample=False,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for pred_text, meta in zip(decoded, batch["meta"]):
                orig = original_data[data_idx]
                predictions.append({
                    "input": orig["input"],
                    "ground_truth": orig["output"],
                    "predicted_output": pred_text,
                    "binary_label": meta["binary_label"],
                    "binary_pred": parse_binary_prediction(pred_text),
                    "meta": {
                        "row_id": meta["row_id"],
                        "question_id": meta["question_id"],
                        "category": meta["category"],
                    },
                })
                data_idx += 1

    return predictions


def train(config: TrainingConfig, device: torch.device, dry_run: bool = False,
          data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR,
          model_output_dir: Path = OUTPUT_DIR):
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Model output: {model_output_dir}")
    print(f"Config: {config}")

    tokenizer = T5Tokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    train_dataset = MisconceptionDataset(data_dir / "train.jsonl", tokenizer, config)
    val_dataset = MisconceptionDataset(data_dir / "val.jsonl", tokenizer, config)
    test_dataset = MisconceptionDataset(data_dir / "test.jsonl", tokenizer, config)

    if dry_run:
        print("DRY RUN: Using tiny subsets...")
        train_dataset.data = train_dataset.data[:16]
        val_dataset.data = val_dataset.data[:8]
        test_dataset.data = test_dataset.data[:8]
        config.epochs = 1

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    training_log = {
        "config": {
            "model_name": config.model_name,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_target_modules": list(config.lora_target_modules),
        },
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "epochs": [],
        "best_epoch": None,
        "best_val_loss": float("inf"),
        "started_at": datetime.now().isoformat(),
    }

    best_val_loss = float("inf")
    model_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'='*60}")

        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimiser.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimiser.step()
            scheduler.step()

            total_train_loss += loss.item()
            num_train_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / num_train_batches

        val_loss, _ = evaluate(model, val_loader, tokenizer, config, device)
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
        }
        training_log["epochs"].append(epoch_log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_log["best_epoch"] = epoch + 1
            training_log["best_val_loss"] = val_loss
            print("New best! Saving checkpoint...")
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)

    # Test predictions using best checkpoint
    print(f"\n{'='*60}")
    print("Generating predictions on test set...")
    print(f"{'='*60}")

    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    model = get_peft_model(model, lora_config)
    model.load_adapter(model_output_dir, adapter_name="default")
    model.to(device)

    predictions = generate_predictions(
        model, test_loader, tokenizer, config, device, test_dataset.data
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = results_dir / "predictions.jsonl"
    with open(predictions_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"Saved {len(predictions)} predictions to {predictions_path}")

    correct = sum(1 for p in predictions if p["binary_label"] == p["binary_pred"])
    accuracy = correct / len(predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    training_log["finished_at"] = datetime.now().isoformat()
    training_log["test_accuracy"] = accuracy
    log_path = results_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Saved training log to {log_path}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f} (epoch {training_log['best_epoch']})")
    print(f"Model saved to: {model_output_dir}")
    print(f"{'='*60}")

    return training_log


def main():
    parser = argparse.ArgumentParser(description="Train T5-small with LoRA")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dry-run", action="store_true", help="Quick test with tiny data")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data directory containing train/val/test.jsonl")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to save model checkpoints")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to save predictions and training log")
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    model_output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    train(config, device, dry_run=args.dry_run, data_dir=data_dir,
          results_dir=results_dir, model_output_dir=model_output_dir)


if __name__ == "__main__":
    main()
