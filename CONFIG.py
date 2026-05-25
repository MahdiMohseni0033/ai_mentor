"""Central configuration for the Mini AI Mentor Engine.

All paths, model names, and tunable parameters live here so the rest of the
code never hard-codes them.  Import the ready-made ``CONFIG`` instance:

    from CONFIG import CONFIG
    print(CONFIG.retrieval.top_k)

Grouped with frozen dataclasses to keep configuration immutable and explicit.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Project root = directory that contains this file.  Everything is relative to it.
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PathConfig:
    root: Path = PROJECT_ROOT

    # Raw inputs
    raw_ted_csv: Path = PROJECT_ROOT / "ted-ultimate-dataset" / "2020-05-01" / "ted_talks_en.csv"
    interviews_dir: Path = PROJECT_ROOT / "data" / "raw" / "interviews"

    # Curated knowledge-base selection (hand-editable source of truth)
    curated_talks_csv: Path = PROJECT_ROOT / "data" / "curated_talks.csv"

    # Processed outputs (produced by the ingestion pipeline)
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    chunks_jsonl: Path = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    documents_csv: Path = PROJECT_ROOT / "data" / "processed" / "documents.csv"
    preprocessing_summary: Path = PROJECT_ROOT / "data" / "processed" / "preprocessing_summary.json"

    # Vector store
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    index_manifest: Path = PROJECT_ROOT / "chroma_db" / "index_manifest.json"

    # Evaluation
    evaluation_dir: Path = PROJECT_ROOT / "data" / "evaluation"
    retrieval_benchmark: Path = PROJECT_ROOT / "data" / "evaluation" / "retrieval_benchmark.json"

    # Prompt templates (markdown, loaded at runtime)
    prompts_dir: Path = PROJECT_ROOT / "prompts"


@dataclass(frozen=True)
class ModelConfig:
    # Local LLM served by Ollama (see system_info.md for the GPU/Ollama setup).
    llm_model: str = "qwen3.6:27b"
    llm_base_url: str = "http://localhost:11434"
    llm_think: bool = False        # qwen3.6 is a "thinking" model; off for fast, clean demo replies
    llm_num_predict: int = 800     # token budget for a grounded mentor answer
    llm_temperature: float = 0.4
    llm_timeout_s: int = 180

    # Embedding model (Qwen3-Embedding; tested on the AMD/ROCm GPU setup).
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    embedding_max_tokens: int = 32768  # model config max_position_embeddings (32K)
    # Qwen3-Embedding uses an instruction prefix for queries (not for documents).
    embedding_query_prompt_name: str = "query"

    # GPU
    use_gpu: bool = True
    gpu_device: str = "7"          # CUDA_VISIBLE_DEVICES value; GPU 8 (index 7) is free on the cluster


@dataclass(frozen=True)
class ChunkingConfig:
    # Paragraph/sentence-aware packing.  Target window with overlap; words, not tokens.
    chunk_size: int = 400          # soft target words per chunk when packing sentences
    chunk_overlap: int = 80        # words of overlap carried into the next chunk
    min_chunk_words: int = 100     # chunks below this get merged into the previous one
    max_chunk_words: int = 600     # hard cap; longer sentence spans are split safely

    # Bump this whenever cleaning/chunking logic changes so the index can detect staleness.
    preprocessing_version: str = "v1"


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 3                 # default number of chunks retrieved per query
    collection_name: str = "ted_mentor"
    # Chroma is configured with cosine space; distance in [0, 2], similarity = 1 - distance.
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class AppConfig:
    page_title: str = "Mini AI Mentor Engine"
    conversation_memory_size: int = 8     # number of recent messages kept as context
    show_retrieved_chunks: bool = True    # display retrieved chunks + scores in the UI
    chunk_preview_chars: int = 280


@dataclass(frozen=True)
class EvaluationConfig:
    num_benchmark_queries: int = 12
    random_seed: int = 42                 # deterministic benchmark sampling
    hit_k_values: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True)
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    app: AppConfig = field(default_factory=AppConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# Single shared instance imported across the project.
CONFIG = Config()


# ---------------------------------------------------------------------------
# Curated knowledge base.
#
# The full TED dataset has ~4,000 talks.  The assignment asks for a small,
# curated set on leadership / career / psychology / productivity.  The curated
# selection (50 talks) lives in data/curated_talks.csv — the hand-editable
# source of truth that the ingestion pipeline reads (see PathConfig.curated_talks_csv).
# Regenerate it with: python scripts/select_curated_talks.py
# ---------------------------------------------------------------------------

# Topics the mentor supports.  Used by the decision engine for off-topic redirects.
SUPPORTED_TOPICS: tuple[str, ...] = (
    "career", "leadership", "communication", "psychology", "productivity",
    "confidence", "purpose", "personal growth",
)
