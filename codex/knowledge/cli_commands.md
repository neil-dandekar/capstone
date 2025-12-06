# CLI Command Reference

Run from repo root:

```bash
cd /Users/neildandekar/Documents/uc-san-diego/dsc180ab/capstone
```

## A) Create/Train CB-LLMs

Use these when you are generating concept labels or training models from scratch/fine-tuning.

### Classification pipeline (build/train)

```bash
python classification/get_concept_labels.py [flags]
python classification/train_CBL.py [flags]
python classification/train_FL.py [flags]
```

Optional LLM labeling route:

```bash
python classification/llm_labeling.py [flags]
python classification/llm_labeling_vllm.py [flags]
```

Baseline training:

```bash
python classification/finetune_black_box.py [flags]
```

### Generation pipeline (build/train)

```bash
python generation/train_CBLLM.py [flags]
python generation/train_classifier.py [flags]
```

## B) Run Pretrained CB-LLMs

Use these when checkpoints already exist in:

- `classification/mpnet_acs/...`
- `generation/from_pretained_llama3_lora_cbm/...`

### 1) Single-text classification inference (pretrained checkpoint)

```bash
python classification/infer_text.py \
  --text "This movie was surprisingly good and emotional." \
  --cbl_path classification/mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt
```

Key flags:

- `--text` required
- `--cbl_path` default: `mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt`
- `--sparse` / `--no-sparse`
- `--max_length` default: `512`
- `--dropout` default: `0.1`
- `--top_k_concepts` default: `8`

### 2) Single-prompt generation inference (pretrained checkpoint)

```bash
python generation/infer_prompt.py \
  --dataset SetFit/sst2 \
  --prompt "Write a short positive movie review:" \
  --max_new_tokens 80
```

Key flags:

- `--prompt` required
- `--dataset` default: `SetFit/sst2`
- `--max_new_tokens` default: `120`
- `--temperature` default: `0.7`
- `--top_k` default: `100`
- `--top_p` default: `0.9`
- `--repetition_penalty` default: `1.5`

### 3) Dataset-level pretrained evaluation commands

Classification:

```bash
python classification/test_CBLLM.py --cbl_path classification/mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt
python classification/print_concept_activations.py --cbl_path classification/mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt
python classification/print_concept_contributions.py --cbl_path classification/mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt
python classification/test_black_box.py --model_path classification/baseline_models/roberta/backbone_finetuned_sst2.pt
```

Generation:

```bash
python generation/test_concepts.py --dataset SetFit/sst2
python generation/test_steerability.py --dataset SetFit/sst2
python generation/test_perplexity.py --dataset SetFit/sst2
python generation/test_weight.py --dataset SetFit/sst2
python generation/test_generation.py --dataset SetFit/sst2
python generation/test_detection.py --dataset yelp_polarity
```

## C) Full Flag Reference

### Classification scripts

`classification/get_concept_labels.py`

- `--dataset` default `SetFit/sst2`
- `--concept_text_sim_model` default `mpnet` (`mpnet|simcse|angle`)
- `--max_length` default `512`
- `--num_workers` default `0`

`classification/llm_labeling.py`

- `--dataset` default `SetFit/sst2`

`classification/llm_labeling_vllm.py`

- `--dataset` default `SetFit/sst2`
- `--tensor_parallel_size` default `4`

`classification/train_CBL.py`

- `--dataset` default `SetFit/sst2`
- `--backbone` default `roberta` (`roberta|gpt2`)
- `--tune_cbl_only` / `--no-tune_cbl_only`
- `--automatic_concept_correction` / `--no-automatic_concept_correction`
- `--labeling` default `mpnet` (`mpnet|angle|simcse|llm`)
- `--cbl_only_batch_size` default `64`
- `--batch_size` default `16`
- `--max_length` default `512`
- `--num_workers` default `0`
- `--dropout` default `0.1`

`classification/train_FL.py`

- `--cbl_path` default `mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt`
- `--batch_size` default `128`
- `--saga_epoch` default `500`
- `--saga_batch_size` default `256`
- `--max_length` default `512`
- `--num_workers` default `0`
- `--dropout` default `0.1`

`classification/test_CBLLM.py`

- `--cbl_path` default `mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt`
- `--sparse` / `--no-sparse`
- `--batch_size` default `256`
- `--max_length` default `512`
- `--num_workers` default `0`
- `--dropout` default `0.1`

`classification/print_concept_activations.py`

- `--cbl_path` default `mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt`
- `--batch_size` default `256`
- `--max_length` default `512`
- `--num_workers` default `0`
- `--dropout` default `0.1`

`classification/print_concept_contributions.py`

- `--cbl_path` default `mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt`
- `--sparse` / `--no-sparse`
- `--batch_size` default `256`
- `--max_length` default `512`
- `--num_workers` default `0`
- `--dropout` default `0.1`

`classification/finetune_black_box.py`

- `--dataset` default `SetFit/sst2`
- `--backbone` default `roberta` (`roberta|gpt2`)
- `--batch_size` default `8`
- `--tune_mlp_only` / `--no-tune_mlp_only`
- `--max_length` default `512`
- `--num_workers` default `0`
- `--projection_dim` default `256`
- `--dropout` default `0.1`

`classification/test_black_box.py`

- `--dataset` default `SetFit/sst2`
- `--model_path` default `baseline_models/roberta/backbone_finetuned_sst2.pt`
- `--batch_size` default `128`
- `--tune_mlp_only` / `--no-tune_mlp_only`
- `--max_length` default `512`
- `--num_workers` default `0`
- `--projection_dim` default `256`
- `--dropout` default `0.1`

`classification/reproduce_table5.py`

- no CLI flags

`classification/infer_text.py`

- `--text` required
- `--cbl_path` default `mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt`
- `--sparse` / `--no-sparse`
- `--max_length` default `512`
- `--dropout` default `0.1`
- `--top_k_concepts` default `8`

### Generation scripts

`generation/train_CBLLM.py`

- `--dataset` default `SetFit/sst2`
- `--batch_size` default `4`
- `--max_length` default `350`
- `--num_workers` default `0`

`generation/train_classifier.py`

- `--dataset` default `SetFit/sst2`
- `--batch_size` default `16`
- `--max_length` default `100`
- `--num_workers` default `0`

`generation/test_concepts.py`

- `--dataset` default `SetFit/sst2`
- `--batch_size` default `8`
- `--max_length` default `350`
- `--num_workers` default `0`

`generation/test_steerability.py`

- `--dataset` default `SetFit/sst2`
- `--max_length` default `1024`

`generation/test_perplexity.py`

- `--dataset` default `SetFit/sst2`
- `--max_length` default `1024`

`generation/test_weight.py`

- `--dataset` default `SetFit/sst2`

`generation/test_generation.py`

- `--dataset` default `SetFit/sst2`
- `--max_length` default `1024`

`generation/test_detection.py`

- `--dataset` default `yelp_polarity`
- `--max_length` default `1024`

`generation/infer_prompt.py`

- `--prompt` required
- `--dataset` default `SetFit/sst2`
- `--max_new_tokens` default `120`
- `--temperature` default `0.7`
- `--top_k` default `100`
- `--top_p` default `0.9`
- `--repetition_penalty` default `1.5`
