from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        if path.suffix.lower() in {'.yaml', '.yml'}:
            return yaml.safe_load(f)
        if path.suffix.lower() == '.json':
            return json.load(f)
    raise ValueError(f'Unsupported config format: {path}')


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
