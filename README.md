# Tibetan spelling correction (v11)

Training and data preparation for Tibetan CSC (`source` → `target`).
Started from [TiSpell](https://arxiv.org/abs/2505.08037); the dataset and
training setup here are ours.

Developed by Dharmaduta for the [Buddhist Digital Resource Center](https://www.bdrc.io/)
(The BDRC Etext Corpus, funded by the Khyentse Foundation).

## Dataset (v11, frozen)

**92,266 pairs** — 82,037 train / 5,146 val / 5,083 test.
Test has **3,839 differing** pairs and **1,244 identical** pairs (`source == target`).

- **Real OCR** — manuscript pages transcribed by annotators. Targets follow the
  page, so they are not always dictionary-correct Tibetan.
- **Synthetic** — BoCorpus and modern news, corrupted on purpose so the target
  is known.

The pair CSVs are **not in this git repo**. Use `../v11_resplit/` or pass
`--data_path` to a folder with `differing_pairs.csv` and `identical_pairs.csv`.

## Model

Default training is **semi-mask**: character and syllable heads, logits summed.
We also tried a copy gate; on the same 3,839-row test it was worse
(correction F1 **0.521** semi-mask vs **0.485** copy-gate). Use `--copy_gate`
only if you want that ablation.

## Layout

```
train.py / infer.py / option.py / metrics.py
model/tispell_roberta.py
dataloader/ocr_pairs.py

scripts/prepare_tispell_data/   # annotator pages → pair CSVs
scripts/data_prep/              # synthetic corrupt + merge + split → v11
```

## Data preparation

```bash
# 1. Real pairs from annotation pages
python3 scripts/prepare_tispell_data/run_pipeline.py \
  --input_dir dataset/ocr_annotation/raw \
  --out_dir dataset/ocr_annotation/exports

# 2. Synthetic pairs
python3 scripts/data_prep/generate_combined.py \
  --bocorpus ../BoCorpus/bo_corpus.parquet \
  --news ../news_data \
  --tispell_corrupt scripts/data_prep/tispell_corrupt \
  --hunspell_dic ../hunspell-bo/bo \
  --out combined_pairs.csv

# 3. Merge real + synthetic
python3 scripts/data_prep/merge_real_v2.py \
  --synthetic combined_pairs.csv \
  --real dataset/ocr_annotation/exports_v2/differing_pairs.csv \
  --tispell_corrupt scripts/data_prep/tispell_corrupt \
  --hunspell_dic ../hunspell-bo/bo \
  --out combined_all.csv

# 4. Split (group by document so pages do not leak across train/test)
python3 scripts/data_prep/split_dataset_v3.py \
  --input combined_all.csv --outdir v11_pairs
```

## Train

```bash
# Linux + CUDA pin. On a Mac: requirements_mac.txt, install torch yourself.
pip install -r requirements.txt

python3 train.py \
  --data_path ../v11_resplit \
  --model_name openpecha/tibetan_RoBERTa_S_e3
```

## Upstream

```
@misc{liu2025tispellsemimaskedmethodologytibetan,
  title={TiSpell: A Semi-Masked Methodology for Tibetan Spelling Correction covering Multi-Level Error with Data Augmentation},
  author={Yutong Liu and Feng Xiao and Ziyue Zhang and Yongbin Yu and Cheng Huang and Fan Gao and Xiangxiang Wang and Ma-bao Ban and Manping Fan and Thupten Tsering and Gadeng Luosang and Renzeng Duojie and Nyima Tashi},
  year={2025},
  eprint={2505.08037},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2505.08037}
}
```
