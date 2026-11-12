"""把 schemas.py 裡的 Pydantic models 匯出成 schema.json。

用法：uv run python scripts/export_schema.py
"""

import json
from pathlib import Path

from creativity3d.schemas import ALL_MODELS

out = {m.__name__: m.model_json_schema() for m in ALL_MODELS}
path = Path(__file__).resolve().parent.parent / "schema.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已匯出 {len(out)} 個 schema → {path.name}")
