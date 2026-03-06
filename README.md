# Opening the Bottleneck: Steering LLMs via Concept Intervention

**Neil Dandekar & Christian Guerra** · UCSD DSC 180B Capstone · Advised by Lily Weng

> **Project Website:** [chguerra15.github.io/capstone-site](https://chguerra15.github.io/capstone-site/) &nbsp;|&nbsp; **Report:** [Q2Report_Checkpoint-3.pdf](./Q2Report_Checkpoint-3.pdf) &nbsp;|&nbsp; **Original Paper:** [CB-LLMs (ICLR 2025)](https://arxiv.org/abs/2412.07992)

---

## Problem Description

Large language models (LLMs) are powerful but opaque — their predictions depend on thousands of hidden features that are not meaningful to humans, making it hard to understand, audit, or correct their behavior.

**Concept Bottleneck Large Language Models (CB-LLMs)** address this by inserting a human-interpretable "concept layer" between the transformer backbone and the final classifier. Each neuron in this layer represents a semantic concept (e.g., sentiment, sports, toxicity), allowing users to inspect and directly manipulate what drives a prediction.

This project reproduces key results from the CB-LLM paper (ICLR 2025) and extends the framework with:
- **Multi-neuron intervention analysis** — coordinated suppression/amplification of concept neuron groups
- **An interactive GUI** — real-time concept manipulation without model retraining
- **Sankey visualizations** — concept weight contribution diagrams across four datasets

---

## Directory Structure

```
capstone/
├── backend_api/          # Flask API connecting the GUI to the CB-LLM model
│   └── app.py            # Main API entrypoint
├── classification/       # CB-LLM text classification pipeline
│   ├── get_concept_labels.py           # Generate concept scores via ACS
│   ├── train_CBL.py                    # Train the Concept Bottleneck Layer
│   ├── train_FL.py                     # Train the sparse linear classifier
│   ├── test_CBLLM.py                   # Evaluate CB-LLM accuracy
│   ├── print_concept_contributions.py  # Generate per-sample explanations
│   ├── finetune_black_box.py           # Train black-box baseline
│   └── requirements.txt
├── generation/           # CB-LLM text generation pipeline
│   ├── train_CBLLM.py          # Finetune Llama3 with CBL
│   ├── test_steerability.py    # Evaluate steerability
│   ├── test_perplexity.py      # Evaluate perplexity
│   ├── test_generation.py      # Generate text with concept intervention
│   ├── test_weight.py          # Extract top concept-token weights for Sankey
│   └── requirements.txt
├── frontend/             # Interactive concept manipulation GUI
│   └── app.py            # Slider-based neuron editor, runs locally
├── codex/                # Additional experiment scripts and analysis
├── fig/                  # Figures used in the report and README
├── checkpoint.ipynb      # Self-contained Q1 reproduction notebook
└── README.md
```

---

## Environment Setup

**Requirements:** CUDA 12.1, Python 3.10, PyTorch 2.2

It is strongly recommended to use a conda virtual environment:

```bash
conda create -n cbllm python=3.10
conda activate cbllm
conda install pytorch==2.2.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## Installation

### Classification Pipeline

```bash
cd classification
pip install -r requirements.txt
```

**Key dependencies:** `transformers>=4.38`, `datasets`, `setfit`, `scikit-learn`, `sentence-transformers`, `tqdm`, `numpy`, `torch==2.2.0`

### Generation Pipeline

```bash
cd generation
pip install -r requirements.txt
```

**Key dependencies:** `transformers>=4.38`, `peft`, `trl`, `accelerate`, `datasets`, `bitsandbytes`, `torch==2.2.0`

### Frontend GUI

```bash
cd frontend
pip install -r requirements.txt
```

---

## Dataset Access

All datasets load automatically via HuggingFace `datasets` — no manual download required. They are cached locally under `~/.cache/huggingface/datasets/` after first run.

| Dataset | HuggingFace ID | Task |
|---|---|---|
| SST-2 | `SetFit/sst2` | Sentiment (binary) |
| Yelp Polarity | `yelp_polarity` | Sentiment (binary) |
| AG News | `ag_news` | Topic (4-class) |
| DBpedia | `dbpedia_14` | Topic (14-class) |

---

## Quickstart: Reproduce Results (No Setup Required)

For Q1 reproduction, open and run `checkpoint.ipynb` — it installs all dependencies automatically and reproduces the key benchmark tables using pretrained checkpoints from HuggingFace.

---

## Running Experiments

### Step 1 — Download Pretrained Checkpoints

Skip training entirely using the authors' finetuned checkpoints:

**Classification:**
```bash
git lfs install
git clone https://huggingface.co/cesun/cbllm-classification temp_repo
mv temp_repo/mpnet_acs classification/
rm -rf temp_repo
```

**Generation:**
```bash
git lfs install
git clone https://huggingface.co/cesun/cbllm-generation temp_repo
mv temp_repo/from_pretained_llama3_lora_cbm generation/
rm -rf temp_repo
```

### Step 2 — Evaluate Classification Accuracy

```bash
cd classification
python test_CBLLM.py --cbl_path mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt --sparse
```

**Expected output:** Per-class and overall test accuracy printed to stdout (~0.96 on SST2). Use `--dataset ag_news`, `--dataset yelp_polarity`, or `--dataset dbpedia_14` to switch datasets.

### Step 3 — Generate Concept Contribution Explanations

```bash
cd classification
python print_concept_contributions.py --cbl_path mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt
```

**Expected output:** 5 concept-based explanations per sample printed to stdout, showing which concepts drove each prediction.

### Step 4 — Visualize Concept Weights (Sankey Diagram)

```bash
cd generation
python test_weight.py --dataset ag_news
```

**Expected output:** `[concept] [token] [weight]` triples printed to stdout. Paste into [SankeyMATIC](https://sankeymatic.com/build/) to generate the flow diagram.

### Step 5 — Evaluate Steerability and Perplexity

```bash
cd generation
python test_steerability.py
python test_perplexity.py
```

**Expected output:** Steerability score (0–1) and perplexity value printed per dataset. Reference values from our reproduction:

| Method | Metric | SST2 | YelpP | AGNews | DBpedia |
|---|---|---|---|---|---|
| CB-LLM (ours) | Accuracy↑ | 0.9638 | 0.9855 | 0.9439 | 0.9924 |
| | Steerability↑ | 0.82 | 0.95 | 0.85 | 0.76 |
| | Perplexity↓ | 116.22 | 13.03 | 18.25 | 37.59 |
| CB-LLM w/o ADV | Accuracy↑ | 0.9676 | 0.9830 | 0.9418 | 0.9934 |
| | Steerability↑ | 0.57 | 0.69 | 0.52 | 0.21 |
| Llama3 (black-box) | Accuracy↑ | 0.9692 | 0.9851 | 0.9493 | 0.9919 |
| | Steerability↑ | No | No | No | No |

### Step 6 — Run Concept Interventions

```bash
cd generation
python test_generation.py --dataset ag_news
```

Edit line 48 of `test_generation.py` to set the neuron activation value you want to intervene on.

**Expected output:** Generated sentences whose topic shifts based on which concept neuron you modified. For example, zeroing the Sports neuron on a sports headline drops predicted Sports probability from 0.89 → 0.48; doubling it raises it to 0.94.

---

## Running the GUI

```bash
cd frontend
python app.py
```

Open the URL shown in the terminal. The interface lets you:
1. Enter any input sentence
2. View the model's concept activations
3. Adjust individual neuron values with sliders
4. See prediction probabilities update in real time

No model retraining is required — the GUI applies scalar multipliers to neuron activations before the classifier head computes logits.

---

## Contribution

| Person | Role |
|---|---|
| **Neil Dandekar** | Backend experimentation, CB-LLM reproduction, intervention logic, model evaluation |
| **Christian Guerra** | Frontend/GUI development, Sankey visualization integration, project website deployment |
| **Lily Weng** | Research advising, methodology guidance, project direction |

---

## Citation

```bibtex
@article{cbllm,
   title={Concept Bottleneck Large Language Models},
   author={Sun, Chung-En and Oikarinen, Tuomas and Ustun, Berk and Weng, Tsui-Wei},
   journal={ICLR},
   year={2025}
}
```

Original authors' repository: [Trustworthy-ML-Lab/CB-LLMs](https://github.com/Trustworthy-ML-Lab/CB-LLMs)
