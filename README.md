# 創造力 3D（Creativity 3D）

選一個詞，寫下聯想，在 3D 星空中看見自己的創造力。規格見 [SPEC.md](SPEC.md)。

> 目前進度：階段 3（環境與架構）。作答與計分功能尚未完成。

## 安裝與執行（Windows）

1. 安裝 uv（若尚未安裝）：`winget install astral-sh.uv`
2. 確認 Ollama 已啟動，並下載模型：
   ```
   ollama pull qwen2.5:7b
   ollama pull bge-m3
   ```
3. 在專案資料夾執行：
   ```
   copy .env.example .env
   uv run creativity3d
   ```
4. 瀏覽器會自動開啟 http://127.0.0.1:8000 ，「系統狀態」顯示綠燈即代表模型連線正常。

沒有 Ollama 時，可在 `.env` 設 `PROVIDER=fake`，用假模型離線開發。

## 架構

```
瀏覽器（frontend/index.html，Three.js）
   │  fetch /api/*
   ▼
FastAPI（app.py）
   ├─ structured.py   Structured Output：JSON Schema → Pydantic 驗證 → 重試 ≤3 → 失敗回 None
   ├─ providers/      LLMProvider / EmbeddingProvider 介面
   │    ├─ ollama.py  本機 Ollama（/api/chat format=schema、/api/embed）
   │    └─ fake.py    測試用假模型
   ├─ schemas.py      所有資料契約（→ schema.json）
   ├─ config.py       讀取 .env
   └─ logging_setup.py 每次 LLM 呼叫寫入 logs/llm.jsonl
```

## 設計理由

- **FastAPI + Three.js，而不是 Streamlit**：3D 互動和網站質感是產品核心，Streamlit 做不到。
- **Provider 介面**：本機用 Ollama 免費；部署到沒有 GPU 的主機時，只要新增一個雲端 API 實作並改 `.env`。
- **LLM 不打分**：分數由程式計算，LLM 只判斷合理性與撰寫解讀，避免分數每次不同或被編造。
