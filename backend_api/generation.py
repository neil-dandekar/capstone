import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaModel


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = REPO_ROOT / "generation"
sys.path.insert(0, str(GENERATION_DIR))

import config as CFG  # noqa: E402
from modules import CBL  # noqa: E402


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(
    prompt: str,
    dataset: str = "SetFit/sst2",
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_k: int = 100,
    top_p: float = 0.9,
    repetition_penalty: float = 1.5,
    top_k_concepts: int = 10,
) -> dict:
    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token
    concept_set = CFG.concepts_from_labels[dataset]

    rel_dir = "from_pretained_llama3_lora_cbm/" + dataset.replace("/", "_")
    peft_path = (GENERATION_DIR / rel_dir / "llama3").resolve()
    cbl_path = (GENERATION_DIR / rel_dir / "cbl.pt").resolve()

    pre_lm = LlamaModel.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16
    ).to(device)
    pre_lm.load_adapter(str(peft_path))
    pre_lm.eval()

    cbl = CBL(config, len(concept_set), tokenizer).to(device)
    cbl.load_state_dict(torch.load(cbl_path, map_location=device))
    cbl.eval()

    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        output_ids, concept_activation = cbl.generate(
            input_ids,
            pre_lm,
            intervene=None,
            length=max_new_tokens,
            temp=temperature,
            topk=top_k,
            topp=top_p,
            repetition_penalty=repetition_penalty,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

    last_step = concept_activation[-1]
    top_k_concepts = max(1, min(top_k_concepts, len(concept_set)))
    values, indices = torch.topk(last_step, k=top_k_concepts)
    top_concepts = []
    for i in range(top_k_concepts):
        idx = int(indices[i].item())
        top_concepts.append(
            {
                "concept_id": concept_set[idx],
                "concept_name": concept_set[idx],
                "activation": float(values[i].item()),
            }
        )

    return {
        "dataset_id": dataset,
        "generated_text": generated_text,
        "full_text": full_text,
        "top_concepts": top_concepts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="SetFit/sst2")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    parser.add_argument("--top_k_concepts", type=int, default=10)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    result = run_inference(
        prompt=args.prompt,
        dataset=args.dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        top_k_concepts=args.top_k_concepts,
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
