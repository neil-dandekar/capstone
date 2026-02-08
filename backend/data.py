from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ResultArtifact:
    key: str
    path: Path
    kind: str  # "json" | "csv"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def results_dir() -> Path:
    return _repo_root() / "results"


def list_artifacts() -> List[ResultArtifact]:
    base = results_dir()
    artifacts: List[ResultArtifact] = []
    if not base.exists():
        return artifacts

    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".json", ".csv"}:
            artifacts.append(ResultArtifact(key=p.stem, path=p, kind=p.suffix.lower().lstrip(".")))
    return artifacts


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path, limit: int = 50_000) -> Dict[str, Any]:
    rows: List[List[Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        columns = next(reader, [])
        for i, r in enumerate(reader):
            if i >= limit:
                break
            rows.append(r)
    return {"columns": columns, "rows": rows}


def get_artifact(key: str) -> Optional[ResultArtifact]:
    for a in list_artifacts():
        if a.key == key:
            return a
    return None


def load_artifact_payload(key: str) -> Any:
    a = get_artifact(key)
    if not a:
        return None
    if a.kind == "json":
        return read_json(a.path)
    if a.kind == "csv":
        return read_csv(a.path)
    return None


def coerce_sankey(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    if "nodes" in payload and "links" in payload:
        if isinstance(payload["nodes"], list) and isinstance(payload["links"], list):
            return payload

    if all(k in payload for k in ("labels", "source", "target", "value")):
        labels = payload["labels"]
        src = payload["source"]
        tgt = payload["target"]
        val = payload["value"]
        if isinstance(labels, list) and isinstance(src, list) and isinstance(tgt, list) and isinstance(val, list):
            nodes = [{"id": str(i), "label": str(l)} for i, l in enumerate(labels)]
            links = [{"source": str(s), "target": str(t), "value": float(v)} for s, t, v in zip(src, tgt, val)]
            return {"nodes": nodes, "links": links}

    if "nodes" in payload and "links" in payload and isinstance(payload["nodes"], list) and all(
        isinstance(x, str) for x in payload["nodes"]
    ):
        nodes = [{"id": str(i), "label": name} for i, name in enumerate(payload["nodes"])]
        links_raw = payload["links"]
        if isinstance(links_raw, list):
            links = []
            for e in links_raw:
                if not isinstance(e, dict):
                    continue
                s = e.get("source", e.get("s"))
                t = e.get("target", e.get("t"))
                v = e.get("value", e.get("v"))
                if s is None or t is None or v is None:
                    continue
                links.append({"source": str(s), "target": str(t), "value": float(v)})
            if links:
                return {"nodes": nodes, "links": links}

    return None
