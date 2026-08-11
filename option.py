import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TiSpell on OCR annotation sentence pairs"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/ocr_annotation/exports",
        help="Directory with differing_pairs.csv and identical_pairs.csv",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openpecha/tibetan_RoBERTa_S_e3",
        help="HuggingFace id or local path for the backbone",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of each split to use (0, 1]; for short experiments",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for --subset sampling",
    )
    parser.add_argument(
        "--identical_ratio",
        type=float,
        default=0.2,
        help=(
            "Target fraction of the train set that is identical pairs "
            "(subsample identical only; val/test keep all)"
        ),
    )
    parser.add_argument(
        "--max_char_length",
        type=int,
        default=380,
        help="Max character length (must match data-prep cap)",
    )
    parser.add_argument(
        "--max_subword_length",
        type=int,
        default=384,
        help="Tokenizer max length (~380 chars at ~1.27 chars/token)",
    )
    parser.add_argument(
        "--max_word_length", type=int, default=64, help="Max length of word"
    )
    parser.add_argument(
        "--max_char_in_word",
        type=int,
        default=12,
        help="Max length of character in a word",
    )

    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=60, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="If >0, stop training after this many optimizer steps (smoke tests)",
    )
    parser.add_argument(
        "--num_epochs_per_evaluation",
        type=int,
        default=1,
        help="Number of epochs per evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default 2e-5; lower than the old 5e-5 to reduce overfitting)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Encoder hidden_dropout_prob and attention_probs_dropout_prob",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early-stop after this many evals without validation-F1 improvement",
    )
    parser.add_argument(
        "--best_checkpoint_dir",
        type=str,
        default="weights/tispell_roberta_best",
        help="Directory for the best-by-val-F1 checkpoint (not overwritten each epoch)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Weight decay"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--w_c",
        type=float,
        default=2.0,
        help="Weight of character-level loss (paper optimal: 2.0)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log metrics to Weights & Biases",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Skip validation (useful for tiny smoke tests)",
    )
    return parser.parse_args()
