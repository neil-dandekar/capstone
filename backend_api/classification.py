import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2TokenizerFast, RobertaModel, RobertaTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_DIR = REPO_ROOT / "classification"
sys.path.insert(0, str(CLASSIFICATION_DIR))

import config as CFG  # noqa: E402
from modules import CBL, GPT2CBL, RobertaCBL  # noqa: E402
from utils import eos_pooling, normalize  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_cbl_path(cbl_path: str) -> tuple[str, str, str, str, Path]:
    p = Path(cbl_path)
    parts = list(p.parts)
    acs_candidates = {"mpnet_acs", "simcse_acs", "angle_acs", "llm_labeling"}

    acs_idx = -1
    for i, part in enumerate(parts):
        if part in acs_candidates:
            acs_idx = i
            break

    if acs_idx < 0 or len(parts) <= acs_idx + 3:
        raise ValueError(
            "cbl_path must contain <acs>/<dataset>/<backbone>/<cbl_file>, "
            "e.g. mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt"
        )

    acs = parts[acs_idx]
    raw_dataset = parts[acs_idx + 1]
    dataset = raw_dataset if "sst2" not in raw_dataset else raw_dataset.replace("_", "/")
    backbone = parts[acs_idx + 2]
    cbl_name = p.name
    cbl_dir = p.parent
    return acs, dataset, backbone, cbl_name, cbl_dir


def _is_intervention_enabled(spec: Any) -> bool:
    return bool(isinstance(spec, dict) and spec.get("enabled"))


def _mechanism(spec: dict) -> str:
    mech = str(spec.get("mechanism") or "activation").lower()
    if mech not in {"activation", "weight", "hybrid"}:
        return "activation"
    return mech


def _apply_activation_ops(
    concept_vec: torch.Tensor,
    concept_to_idx: dict[str, int],
    spec: dict,
) -> torch.Tensor:
    """
    concept_vec: [num_concepts] on CPU
    spec: UI sends spec["activation"]["per_concept"][concept_id] = {op, value, relu_after}
    """
    out = concept_vec.clone()
    per = (spec.get("activation") or {}).get("per_concept") or {}
    if not isinstance(per, dict):
        return out

    for concept_id, rule in per.items():
        if concept_id not in concept_to_idx:
            continue
        if not isinstance(rule, dict):
            continue

        j = concept_to_idx[concept_id]
        op = str(rule.get("op") or "add").lower()
        try:
            val = float(rule.get("value") or 0.0)
        except Exception:
            val = 0.0
        relu_after = bool(rule.get("relu_after") or False)

        if op == "add":
            out[j] = out[j] + val
        elif op == "override":
            out[j] = torch.tensor(val, dtype=out.dtype)
        elif op == "scale":
            out[j] = out[j] * val
        # unknown ops are ignored

        if relu_after:
            out[j] = torch.relu(out[j])

    return out


def _apply_weight_ops(
    W: torch.Tensor,
    concept_to_idx: dict[str, int],
    spec: dict,
) -> torch.Tensor:
    """
    W: [num_classes, num_concepts] on CPU
    spec: UI sends spec["weight"]["per_concept"][concept_id] = {action, value}
    action: "zero" -> W[:, j] = 0
    action: "scale" -> W[:, j] *= value
    """
    out = W.clone()
    per = (spec.get("weight") or {}).get("per_concept") or {}
    if not isinstance(per, dict):
        return out

    for concept_id, rule in per.items():
        if concept_id not in concept_to_idx:
            continue
        if not isinstance(rule, dict):
            continue

        j = concept_to_idx[concept_id]
        action = str(rule.get("action") or "scale").lower()
        try:
            val = float(rule.get("value") or 0.0)
        except Exception:
            val = 0.0

        if action == "zero":
            out[:, j] = 0.0
        elif action == "scale":
            out[:, j] = out[:, j] * val
        # unknown actions ignored

    return out


def run_inference(
    text: str,
    cbl_path: str,
    top_k_concepts: int = 8,
    max_length: int = 512,
    dropout: float = 0.1,
    sparse: bool = False,
    intervention: dict | None = None,
) -> dict:
    cbl_path_obj = Path(cbl_path)
    if not cbl_path_obj.is_absolute():
        cbl_path_obj = CLASSIFICATION_DIR / cbl_path_obj
    cbl_path_obj = cbl_path_obj.resolve()

    _, dataset, backbone, cbl_name, cbl_dir = parse_cbl_path(str(cbl_path_obj))
    concept_set = CFG.concept_set[dataset]
    concept_to_idx = {c: i for i, c in enumerate(concept_set)}

    if "roberta" in backbone:
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        if "no_backbone" in cbl_name:
            cbl = CBL(len(concept_set), dropout).to(device)
            cbl.load_state_dict(torch.load(cbl_path_obj, map_location=device))
            cbl.eval()
            pre_lm = RobertaModel.from_pretrained("roberta-base").to(device)
            pre_lm.eval()
            backbone_cbl = None
        else:
            backbone_cbl = RobertaCBL(len(concept_set), dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(cbl_path_obj, map_location=device))
            backbone_cbl.eval()
            cbl = None
            pre_lm = None
    elif "gpt2" in backbone:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        if "no_backbone" in cbl_name:
            cbl = CBL(len(concept_set), dropout).to(device)
            cbl.load_state_dict(torch.load(cbl_path_obj, map_location=device))
            cbl.eval()
            pre_lm = GPT2Model.from_pretrained("gpt2").to(device)
            pre_lm.eval()
            backbone_cbl = None
        else:
            backbone_cbl = GPT2CBL(len(concept_set), dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(cbl_path_obj, map_location=device))
            backbone_cbl.eval()
            cbl = None
            pre_lm = None
    else:
        raise ValueError("Backbone must be roberta or gpt2 based on cbl_path.")

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        if "no_backbone" in cbl_name:
            features = pre_lm(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).last_hidden_state
            if "roberta" in backbone:
                features = features[:, 0, :]
            else:
                features = eos_pooling(features, batch["attention_mask"])
            concept_features = cbl(features)
        else:
            concept_features = backbone_cbl(batch)

    # concept_features: [1, num_concepts]
    concept_features = concept_features.detach().cpu()

    model_name = cbl_name[3:]
    train_mean = torch.load(cbl_dir / ("train_mean" + model_name), map_location="cpu")
    train_std = torch.load(cbl_dir / ("train_std" + model_name), map_location="cpu")
    concept_features, _, _ = normalize(
        concept_features, d=0, mean=train_mean, std=train_std
    )
    concept_features = F.relu(concept_features)

    # Load linear head (W, b) on CPU
    final = torch.nn.Linear(
        in_features=len(concept_set), out_features=CFG.class_num[dataset]
    )
    w_name = "W_g"
    b_name = "b_g"
    if sparse:
        w_name += "_sparse"
        b_name += "_sparse"

    w = torch.load(cbl_dir / (w_name + model_name), map_location="cpu")
    b = torch.load(cbl_dir / (b_name + model_name), map_location="cpu")

    # --- apply interventions (classification) ---
    if _is_intervention_enabled(intervention):
        mech = _mechanism(intervention)

        # Activation intervention modifies concept_features (vector)
        if mech in {"activation", "hybrid"}:
            concept_features[0] = _apply_activation_ops(
                concept_features[0], concept_to_idx, intervention
            )

        # Weight intervention modifies outgoing weights in head: columns of W
        if mech in {"weight", "hybrid"}:
            w = _apply_weight_ops(w, concept_to_idx, intervention)

    final.load_state_dict({"weight": w, "bias": b})

    with torch.no_grad():
        logits = final(concept_features)[0]
        probs = F.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs).item())

    class_names = CFG.concepts_from_labels[dataset]
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    top_k = max(1, min(top_k_concepts, len(concept_set)))
    values, indices = torch.topk(concept_features[0], k=top_k)

    top_concepts = []
    for i in range(top_k):
        idx = int(indices[i].item())
        top_concepts.append(
            {
                "concept_id": concept_set[idx],
                "concept_name": concept_set[idx],
                "activation": float(values[i].item()),
            }
        )

    probabilities = {}
    logits_map = {}
    for i, name in enumerate(class_names):
        if i >= len(probs):
            break
        probabilities[name] = float(probs[i].item())
        logits_map[name] = float(logits[i].item())

    return {
        "dataset_id": dataset,
        "predicted_label": pred_label,
        "prediction_index": pred_idx,
        "confidence": float(probs[pred_idx].item()),
        "probabilities": probabilities,
        "logits": logits_map,
        "top_concepts": top_concepts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument(
        "--cbl_path",
        type=str,
        default="mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt",
    )
    parser.add_argument("--sparse", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--top_k_concepts", type=int, default=8)
    parser.add_argument(
        "--intervention_json",
        type=str,
        default="",
        help="JSON string specifying interventions (UI payload.intervention).",
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    intervention = None
    if args.intervention_json:
        try:
            intervention = json.loads(args.intervention_json)
        except json.JSONDecodeError:
            intervention = None

    result = run_inference(
        text=args.text,
        cbl_path=args.cbl_path,
        top_k_concepts=args.top_k_concepts,
        max_length=args.max_length,
        dropout=args.dropout,
        sparse=bool(args.sparse),
        intervention=intervention,
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
