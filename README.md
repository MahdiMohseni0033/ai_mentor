# 🧭 Mini AI Mentor Engine

A small, transparent **RAG mentor** that answers career / leadership / psychology /
productivity questions, grounded in a curated set of TED talks. It classifies
each query, retrieves relevant excerpts from a vector store, and generates a
warm, practical, **source-grounded** mentor reply with a local LLM.

Built to be **simple, testable, and easy to explain** — not a multi-agent
framework.

---

## 1. Project overview

Given a user question, the system:

1. **Routes** the query (emotional / strategic / vague / general / off-topic,
   plus conversational follow-ups and appreciation) with a context-aware **LLM
   turn router** — role-modelled, XML-structured, few-shot — that degrades to a
   deterministic rule-based engine if the LLM is unavailable.
2. **Decides whether RAG is needed** before embedding anything, reading the
   conversation, not keywords. "Translate that to French" is answered from
   history; a pushy follow-on to an off-topic question stays refused; "thanks"
   gets a short reply; real mentor questions retrieve.
3. **Retrieves** the top-k most relevant talk excerpts from ChromaDB.
4. **Generates** a structured mentor response with a local LLM (Ollama),
   grounded only in the retrieved excerpts.
5. **Shows its work** — the classification, the retrieved chunks, and their
   cosine-similarity scores are all surfaced in the UI.

Conversation memory lets it handle follow-ups (e.g. "Help" → clarifying
question → "mostly career" → full grounded answer using the earlier context).

## 2. Architecture summary

```
OFFLINE (build once)                      ONLINE (per query)
─────────────────────                     ──────────────────
raw transcripts                           user query
   │ clean (deterministic)                   │ route (LLM router)
   │ chunk (paragraph/sentence-aware)        │ embed query
   │ embed (Qwen3-Embedding-0.6B)            │ retrieve top-k (Chroma, cosine)
   ▼                                         │ build grounded prompt
ChromaDB  ◀── smart resume/upsert            ▼ call LLM (Ollama, qwen3.6:27b)
                                          structured mentor answer + sources
```

A single **`MentorController`** orchestrates classify → retrieve → generate.
See [`architecture.md`](architecture.md) for design decisions and trade-offs.

## 3. Repository structure

```
ai_mentor/
├── app.py                     # Streamlit demo (browser UI)         ← entry point
├── chat_cli.py                # terminal chat demo (no browser)      ← entry point
├── CONFIG.py                  # all config as dataclasses + curated talk list
├── conftest.py                # pytest setup (path + shared fixtures)
├── requirements.txt
├── README.md / architecture.md / agent.md / CLAUDE.md   # docs
├── system_info.md             # authoritative cluster / Ollama / GPU facts
├── start_ollama_gpu.sh        # start the local LLM on the GPU node
│
├── prompts/                   # editable prompts: system_prompt.md, router_prompt.md
│
├── data/
│   ├── raw/interviews/        # optional interview JSONs (see its README)
│   ├── processed/             # chunks.jsonl, documents.csv, summary  (generated)
│   └── evaluation/            # retrieval_benchmark.json               (generated)
├── chroma_db/                 # persistent vector index + manifest      (generated)
│
├── src/
│   ├── controller.py          # the single orchestration layer
│   ├── ingestion/             # load_transcripts, clean_text, chunk_text, pipeline
│   ├── embeddings/            # embedding_model (Qwen3-Embedding wrapper)
│   ├── vectorstore/           # chroma_store (upsert + resume helpers)
│   ├── retrieval/             # retriever (query → top-k chunks + scores)
│   ├── decision/              # turn_router (LLM) + query_classifier (rule fallback)
│   ├── generation/            # llm_client + mentor_response (prompt/persona)
│   ├── evaluation/            # build_retrieval_benchmark + evaluate_retrieval
│   └── utils/                 # text_utils, prompts, healthcheck
│
├── scripts/                   # helper scripts: build_index, run_healthcheck,
│                              # run_demo_queries, select_curated_talks,
│                              # download_dataset, amd_gpu_brief.sh
├── tests/                     # pytest suite
└── report/                    # 2-page LaTeX delivery report + compile script
```

## 4. Setup

```bash
ssh gpu123                                   # the AMD-GPU node (see system_info.md)
cd /mnt/scratch2/users/mmohseni/projects/ai_mentor
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Model setup

This project uses two **local** models — nothing is sent to the cloud.

* **LLM — `qwen3.6:27b` via [Ollama](https://ollama.com).** Start it on the free
  GPU (see `system_info.md`); GPU 8 / index `7` is used by default:
  ```bash
  bash start_ollama_gpu.sh          # starts `ollama serve` with ROCm env vars
  ollama run qwen3.6:27b            # ensure the model is pulled/loaded
  ```
  `qwen3.6` is a *thinking* model; we call it with `think=False` for fast, clean
  replies (configurable in `CONFIG.py`).

* **Embeddings — `Qwen/Qwen3-Embedding-0.6B`** (1024-dim, 32K token model
  context) via `sentence-transformers`, loaded on the GPU with `bfloat16` +
  `sdpa`, using the tested AMD/ROCm GPU setup.

* **AMD GPU notes:** the cluster runs ROCm (`torch==2.12.0+rocm7.2`). The GPU is
  selected with `CUDA_VISIBLE_DEVICES` (default `7`, set in `CONFIG.py`). See
  **`system_info.md`** for the authoritative cluster/Ollama/GPU details (that
  file is not modified by this project). `bash scripts/amd_gpu_brief.sh` shows GPU usage.

## 6–7. Data pipeline & index

```bash
# (only if the raw dataset is missing) download the TED dataset from Kaggle
python scripts/download_dataset.py

# clean + chunk the curated talks -> data/processed/
python -m src.ingestion.pipeline

# embed + index into ChromaDB (smart resume: skips unchanged chunks)
python scripts/build_index.py          # use --rebuild to force re-embedding
```

The pipeline curates **50 on-theme talks** (Simon Sinek, Brené Brown, Amy Cuddy,
Tim Urban, Angela Duckworth, and others) out of the full 4,000-talk dataset,
producing **358 clean chunks** in the current processed build. The curated list
lives in `data/curated_talks.csv`.

## 8. Run the demo

**Streamlit (browser UI):**

```bash
streamlit run app.py
```

The UI provides a chat box, example queries, the decision label + reason,
retrieved chunks with similarity scores (expandable), a config sidebar, and a
clear-conversation button.

**Terminal (interactive chat) — same functionality, no browser:**

```bash
python chat_cli.py
```

An interactive REPL with conversation memory that prints the decision label +
reason, the grounded answer, and the retrieved chunks with similarity scores.
Commands: `/sources`, `/examples`, `/clear`, `/help`, `/exit`.

For a quick non-interactive smoke test over a few fixed queries:

```bash
python scripts/run_demo_queries.py
```

## 9. Tests

```bash
pytest -q
```

Covers cleaning, chunking, the decision engine, controller routing, the prompt
builder, and (if the index + embedding model are available) real embeddings and
retrieval. Tests that need the heavy model skip gracefully when it can't load.

## 10. Retrieval benchmark

```bash
python -m src.evaluation.build_retrieval_benchmark   # writes data/evaluation/retrieval_benchmark.json
python -m src.evaluation.evaluate_retrieval          # prints hit@k / MRR
```

The builder asks the LLM to *paraphrase* a query for a deterministically-sampled
chunk per talk (fixed seed), saving `query → expected_chunk_id/doc_id`. It is
written once and reused (`--force` to regenerate; manual fallback if the LLM is
down). The evaluator reports chunk-level and doc-level **hit@1/3/5 + MRR**.

## 11. Example queries

| Query | Classified as |
|---|---|
| "I feel stuck in my career and don't know what to do next." | emotional |
| "How can I become a better leader?" | strategic |
| "Give me a 7-day plan to improve focus." | strategic |
| "Help" | vague → asks a clarifying question |
| "How do I find my purpose?" | strategic |
| "What is the capital of France?" | off_topic → polite redirect |
| (after an answer) "summarize that in 2 sentences" | follow_up → answered from history, no new retrieval |
| (after an answer) "that was great" | appreciation → short social reply, no retrieval |
| (after an answer) "liked it" | appreciation → short social reply, no retrieval |

## 12. Example output shape

```
💙 emotional — Contains emotional terms (feel, stuck).

**Reflection** — It's completely normal to feel stuck when the path isn't clear...
**Mentor insight** — Larry Smith argues we often fail not from lack of talent but
                     from the excuses we make to avoid our real passions...
**Practical steps** — 1) Name your excuses  2) Redefine "weird"  3) ...
**One small next step** — Spend ten minutes writing three things you're curious about.

Retrieved context:
  🔎 0.586 · Larry Smith — "Why you will fail to have a great career"  (ted_1384_chunk_000)
  🔎 0.522 · Larry Smith — ...
```

## 13. Assumptions & limitations

* **TED-only dataset.** The local dataset has no interviews, so the demo uses TED
  talks. Interview support exists (drop JSONs in `data/raw/interviews/`, see its
  README) — we don't fabricate sources. (Bill Gates *is* in the dataset but his
  talks are climate/health, off-theme, so they're excluded.)
* **Curated subset.** 50 talks keep the index small and the demo fast; scaling up
  is just adding rows to `data/curated_talks.csv` and re-running the pipeline.
* **Latency.** A grounded LLM answer takes ~10–17s on the local GPU; vague/off-topic
  replies are instant (templated, no LLM).
* **Embedding similarity** for Qwen3-Embedding sits in a moderate cosine range
  (~0.4–0.65 for good matches); scores are relative, not absolute "confidence".

## 14. Future improvements

* Lightweight reranking (cross-encoder) to lift chunk-level hit@1.
* Streaming responses for snappier UX.
* A larger curated set + per-topic retrieval filters.
* An answer-grounding/faithfulness check (LLM-as-judge) added to the benchmark.
* Promote the embedding model to a small FastAPI service to warm-load it across
  processes (kept in-process here for demo simplicity — see architecture.md).

---

### Quick start (TL;DR)

```bash
source .venv/bin/activate && pip install -r requirements.txt
bash start_ollama_gpu.sh           # (separate terminal) + `ollama run qwen3.6:27b`
python scripts/run_healthcheck.py  # verify LLM + embeddings + index
python -m src.ingestion.pipeline && python scripts/build_index.py   # if not built
streamlit run app.py
```

See [`architecture.md`](architecture.md) for the design and the **demo script**.
