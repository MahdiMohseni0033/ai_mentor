"""Embedding model wrapper around Qwen3-Embedding-0.6B.

Uses the tested AMD/ROCm GPU setup (bfloat16 + sdpa).
The model is loaded once and reused; in Streamlit it is cached with
`st.cache_resource`, and the offline index build loads it a single time.

Vectors are L2-normalized so cosine similarity == dot product, which matches
the cosine space configured on the Chroma collection.
"""

from __future__ import annotations

import os

from CONFIG import CONFIG


class EmbeddingModel:
    def __init__(
        self,
        model_name: str | None = None,
        use_gpu: bool | None = None,
        gpu_device: str | None = None,
    ):
        self.model_name = model_name or CONFIG.model.embedding_model
        use_gpu = CONFIG.model.use_gpu if use_gpu is None else use_gpu
        gpu_device = gpu_device or CONFIG.model.gpu_device

        # CRITICAL: select the GPU before torch / sentence_transformers import.
        if use_gpu:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_device)

        from sentence_transformers import SentenceTransformer  # lazy import

        model_kwargs: dict = {"attn_implementation": "sdpa"}
        if use_gpu:
            model_kwargs["device_map"] = "auto"      # maps to the single visible GPU
            model_kwargs["torch_dtype"] = "bfloat16"

        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs=model_kwargs,
            processor_kwargs={"padding_side": "left"},
        )
        self._query_prompt = CONFIG.model.embedding_query_prompt_name

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages/chunks (no query instruction prefix)."""
        vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (uses Qwen3's query instruction prompt)."""
        vector = self.model.encode(
            [text],
            prompt_name=self._query_prompt,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()
