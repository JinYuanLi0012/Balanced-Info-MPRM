# Training Data Efficiency in Multimodal Process Reward Models

> Make Multimodal Process Reward Model (MPRM) training 10× cheaper by selecting informative Monte Carlo (MC)-annotated rollouts—no extra supervision, no extra model calls.

## 🔥 Updates

- **[2026-02-04]** We released our [paper](https://arxiv.org/abs/2602.11111) and [code](https://github.com/JinYuanLi0012/Balanced-Info-MPRM). BIS can match full-data performance using as little as **10%** of the training data.

## 🧩 Overview
Training MPRMs usually relies on large-scale MC-annotated corpora, which makes training expensive. Our study shows that random subsampling saturates quickly, implying strong redundancy in existing MC rollouts. 

We identify two key factors that determine whether a rollout yields informative gradient updates:

**Label Mixture**: rollouts that include both positive and negative steps (i.e., “mixed”) provide stronger supervision signals. 

**Label Reliability**: positive steps with higher average MC scores are more reliable; extremely low-MC positives are often noisy pseudo-positives. 

Built on empirical observations and grounded analysis, we propose the Balanced-Information Score (BIS) to rank rollouts by mixture × reliability, using only existing MC signals stored in the dataset.

## ⚡️ Quickstart Guide

### 1. Configure Environment
```bash
git clone https://github.com/JinYuanLi0012/Balanced-Info-MPRM.git
cd Balanced-Info-MPRM

conda create -n Balanced-Info-MPRM python=3.10 -y
conda activate Balanced-Info-MPRM

pip install uv
uv pip install -r requirements.txt
```

### 2. Data Prepare

We use the MC-annotated VisualPRM400K dataset:

- Hugging Face: `OpenGVLab/VisualPRM400K-v1.1-Raw`

Download the dataset from Hugging Face and place it under:

${PROJECT_ROOT}/datasets/VisualPRM400K-v1.1-raw/

```bash
datasets/VisualPRM400K-v1.1-raw/
  -- annotations/                # original VisualPRM400K annotations (=38 .jsonl files)
    -- ai2d_train_12k_en_20240410_extracted.jsonl
    -- chartqa_trainval_30k_w_csv_en_20240402_extracted.jsonl
    ...
  -- VisualPRM400K-v1.1-Raw/     # image folders (=38 folders)
    -- ai2d
    -- chartqa
    ...
```

### 3. Full-data and BIS-selected Training sets Process

#### Full-data (100%)
1) Convert raw VisualPRM400K annotations to PRM hard-label jsonl (38 files)
```bash
python scripts/convert_visualprm400k_to_mmprm.py
```

2) Merge all hard-label PRM files into one full training set
```bash
cd ${PROJECT_ROOT}/datasets/VisualPRM400K-v1.1-raw
cat converted_hard/*_prm.jsonl > all_combined_data_hard.jsonl
```

#### BIS subset
> **BIS hyperparameters**: `--top-ratio` is the fraction of rollouts kept from each annotation file (e.g., `0.25` keeps the top 25%). `--alpha` is the BIS smoothing hyperparameter in \((p_{pos}(1-p_{pos})+\alpha)\cdot R\). In practice, when using a **smaller** `--top-ratio` (more aggressive filtering), it can help to set a **slightly larger** `--alpha` for more stable selection.

```bash
cd ${PROJECT_ROOT}

python scripts/build_bis_subset.py \ # 10% subset
  --annotations-dir datasets/VisualPRM400K-v1.1-raw/annotations \
  --alpha 0.2 \
  --top-ratio 0.1 \
  --output datasets/VisualPRM400K-v1.1-raw/bis10_alpha0_2_combined_data_hard_prm.jsonl

python scripts/build_bis_subset.py \ # 25% subset
  --annotations-dir datasets/VisualPRM400K-v1.1-raw/annotations \
  --alpha 0.05 \
  --top-ratio 0.25 \
  --output datasets/VisualPRM400K-v1.1-raw/bis25_alpha0_05_combined_data_hard_prm.jsonl
```

After running, you should have:

```bash
datasets/VisualPRM400K-v1.1-raw/
  -- all_combined_data_hard.jsonl                     # full PRM training data
  -- bis10_alpha0_2_combined_data_hard_prm.jsonl      # BIS-subsets (10%) training data
  -- bis25_alpha0_05_combined_data_hard_prm.jsonl     # BIS-subsets (25%) training data
  -- converted_hard/
    -- ai2d_train_12k_en_20240410_extracted_prm.jsonl
    -- chartqa_trainval_30k_w_csv_en_20240402_extracted_prm.jsonl
    ...
```

### 4. Model Training
We provide a ready-to-run training script `shell/scripts/visualprm400k_train.sh`.
By default, it reads the dataset meta config from `shell/data/meta_visualprm400k.json`
and trains an InternVL2.5 model with DeepSpeed ZeRO-3 (`configs/zero_stage3_config.json`).

The default parameters are suitable for 4 GPUs with at least 80GB of memory.

> To train on a different dataset/split, edit the `annotation` field in `shell/data/meta_visualprm400k.json`
> (e.g., switch from `.../all_combined_data_hard.jsonl` to a BIS subset jsonl).

```bash
bash shell/scripts/visualprm400k_train.sh
```

### 5. Evaluation

We provide the VisualProcessBench evaluation script:

- `eval/prm/evaluate_visualprocessbench_prm_new.py`

Assume VisualProcessBench is placed as:

```bash
datasets/VisualProcessBench/
  -- test.jsonl
  -- images         # VisualProcessBench images referenced by `test.jsonl`
```

Run evaluation (single or multi-GPU):

```bash
cd ${PROJECT_ROOT}   # ${PROJECT_ROOT} = Bananced-Info/
export PYTHONPATH="$(pwd)/src"

GPUS=4
CKPT_DIR="/path/to/your/checkpoint"            # e.g., work_dirs/.../checkpoint-50
ANN="datasets/VisualProcessBench/test.jsonl"
IMG_ROOT="datasets/VisualProcessBench"
OUT_DIR="${CKPT_DIR}/eval_visualprocessbench"

torchrun --nproc_per_node=${GPUS} eval/prm/evaluate_visualprocessbench_prm_new.py \
  --checkpoint "${CKPT_DIR}" \
  --annotation "${ANN}" \
  --image-root "${IMG_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --auto \
  --dynamic \
  --max-num 6
```
