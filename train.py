import os
import re
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler, AutoTokenizer
from tqdm import tqdm

from option import parse_args
from dataloader.ocr_pairs import OCRAnnotationDataset
from metrics import compute_precision_recall_f1
from model.tispell_roberta import TiSpell_RoBERTa

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def _mean_token_loss(logits, targets, attention_mask, pad_id: int):
    """Per-sample mean CE over non-pad tokens. logits: [B, L, V], targets: [B, L]."""
    bsz, seqlen, vocab = logits.shape
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        reduction="none",
        ignore_index=pad_id,
    ).view(bsz, seqlen)
    # Prefer gold attention mask; fall back to non-pad on targets.
    if attention_mask is None:
        mask = (targets != pad_id).float()
    else:
        mask = attention_mask.float()
        mask = mask * (targets != pad_id).float()
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (loss * mask).sum(dim=1) / denom


def train(args, model, train_loader, optimizer, lr_scheduler, device, pad_id: int):
    model.train()
    total_loss = 0.0
    n_steps = 0
    max_steps = getattr(args, "max_train_steps", 0) or 0

    for batch in tqdm(train_loader, desc="Training"):
        target_input_ids = batch["target"]["input_ids"].to(device)
        target_attention = batch["target"]["attention_mask"].to(device)
        source_input_ids = batch["random_corrupt"]["input_ids"].to(device)
        source_attention_mask = batch["random_corrupt"]["attention_mask"].to(device)
        target_tag_ids = batch["mask"]["input_ids"].to(device)

        logit, logit_c = model(
            source_input_ids, attention_mask=source_attention_mask
        )

        loss_tok = _mean_token_loss(
            logit, target_input_ids, target_attention, pad_id
        )
        loss_c_tok = _mean_token_loss(
            logit_c, target_tag_ids, target_attention, pad_id
        )
        loss = (loss_tok + args.w_c * loss_c_tok).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        n_steps += 1
        if max_steps > 0 and n_steps >= max_steps:
            break

    avg_loss = total_loss / max(n_steps, 1)
    return avg_loss, n_steps


def validation(model, test_loader, tokenizer, device, epoch, results_per_epoch):
    """Evaluate on real pairs; report overall and per-diff_category P/R/F1."""
    model.eval()
    buckets = defaultdict(lambda: {"pre": 0.0, "rec": 0.0, "f1": 0.0, "count": 0})

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            target_input_ids = batch["target"]["input_ids"].to(device)
            source_input_ids = batch["random_corrupt"]["input_ids"].to(device)
            source_attention_mask = batch["random_corrupt"]["attention_mask"].to(
                device
            )
            categories = batch["diff_category"]

            target_texts = tokenizer.batch_decode(
                target_input_ids, skip_special_tokens=True
            )
            logit, _ = model(
                source_input_ids, attention_mask=source_attention_mask
            )
            pred_input_ids = torch.argmax(logit, dim=-1)
            pred_texts = tokenizer.batch_decode(
                pred_input_ids, skip_special_tokens=True
            )

            for pred_text, target_text, cat in zip(
                pred_texts, target_texts, categories
            ):
                # Syllable-level metric (tsheg as separator), matching original.
                pred_syl = re.split(r"[་༌]", pred_text.replace(" ", ""))
                gold_syl = re.split(r"[་༌]", target_text.replace(" ", ""))
                pred_syl = [s for s in pred_syl if s]
                gold_syl = [s for s in gold_syl if s]
                precision, recall, f1 = compute_precision_recall_f1(
                    pred_syl, gold_syl
                )
                for key in (cat, "overall"):
                    buckets[key]["pre"] += precision
                    buckets[key]["rec"] += recall
                    buckets[key]["f1"] += f1
                    buckets[key]["count"] += 1

    # Stable print order: overall first, then categories by count.
    ordered = sorted(
        buckets.items(),
        key=lambda kv: (0 if kv[0] == "overall" else 1, -kv[1]["count"], kv[0]),
    )
    results = {}
    print("\nValidation (precision / recall / F1):")
    for name, metrics in ordered:
        if metrics["count"] == 0:
            continue
        n = metrics["count"]
        pre = metrics["pre"] / n
        rec = metrics["rec"] / n
        f1 = metrics["f1"] / n
        results[name] = {"pre": pre, "rec": rec, "f1": f1, "count": n}
        print(
            f"  {name:<28} n={n:5d}  "
            f"pre={pre:.4f}  rec={rec:.4f}  f1={f1:.4f}"
        )

    results_per_epoch[epoch] = results
    if wandb is not None and wandb.run is not None:
        for key, metrics in results.items():
            for metric_name, value in metrics.items():
                if metric_name != "count":
                    wandb.log({f"val/{key}_{metric_name}": value}, step=epoch)

    os.makedirs("weights/tispell_roberta", exist_ok=True)
    model.roberta.save_pretrained("weights/tispell_roberta")
    tokenizer.save_pretrained("weights/tispell_roberta")
    torch.save(model.state_dict(), "weights/tispell_roberta/tispell_roberta.pth")
    return results_per_epoch


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed; omit --use_wandb or install it")
        wandb.init(
            project="TiSpell",
            name="tispell_roberta_ocr",
            config={
                "model": args.model_name,
                "language": "Tibetan",
                "hidden_size": args.hidden_size,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "subset": args.subset,
                "identical_ratio": args.identical_ratio,
                "max_char_length": args.max_char_length,
                "max_subword_length": args.max_subword_length,
                "w_c": args.w_c,
            },
        )

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        # RoBERTa-style: reuse eos as pad if needed.
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    pad_id = tokenizer.pad_token_id

    model = TiSpell_RoBERTa(args.model_name, tokenizer)
    model.to(device)

    train_dataset = OCRAnnotationDataset(args, tokenizer, split="train")
    val_dataset = OCRAnnotationDataset(args, tokenizer, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    if args.max_train_steps and args.max_train_steps > 0:
        num_training_steps = args.max_train_steps
    else:
        num_training_steps = max(steps_per_epoch * args.epochs, 1)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    results_per_epoch = {}
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, n_steps = train(
            args, model, train_loader, optimizer, lr_scheduler, device, pad_id
        )
        print(f"Training loss: {train_loss:.4f}  steps={n_steps}")
        if args.use_wandb and wandb is not None:
            wandb.log({"train/loss": train_loss}, step=epoch)

        if not args.no_eval and (epoch + 1) % args.num_epochs_per_evaluation == 0:
            results_per_epoch = validation(
                model, val_loader, tokenizer, device, epoch, results_per_epoch
            )

        if args.max_train_steps and args.max_train_steps > 0:
            # Smoke / short runs: one partial epoch is enough.
            break

    if args.use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
