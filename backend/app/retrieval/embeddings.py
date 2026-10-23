"""Embedding providers.

* ``HashingEmbedder``: local and deterministic (feature hashing of words, word
  bigrams and character n-grams). No model download and no API key, so the
  public demo always works. It captures lexical similarity, not deep semantics.
* ``OpenAIEmbedder``: any OpenAI-compatible ``/embeddings`` endpoint
  (OpenAI, OpenRouter, a local server...).
* ``AzureOpenAIEmbedder``: an Azure OpenAI embeddings deployment.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import httpx
import numpy as np

from app.retrieval.text import tokenize


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self.dim, sign

    def _features(self, text: str) -> list[tuple[str, float]]:
        tokens = tokenize(text)
        features: list[tuple[str, float]] = [(f"w:{t}", 1.0) for t in tokens]
        features += [(f"b:{a}_{b}", 0.7) for a, b in zip(tokens, tokens[1:], strict=False)]
        for token in tokens:  # char 4-grams help with inflections and typos
            padded = f"#{token}#"
            features += [(f"c:{padded[i:i + 4]}", 0.3) for i in range(max(1, len(padded) - 3))]
        return features

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                index, sign = self._bucket(feature)
                matrix[row, index] += sign * weight
        # sublinear term frequency, keeping the sign
        matrix = np.sign(matrix) * np.log1p(np.abs(matrix))
        return _normalize_rows(matrix)


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str, batch_size: int = 64):
        if not api_key:
            raise ValueError("CITADOC_OPENAI_API_KEY is required for OpenAI embeddings")
        self.model = model
        self.batch_size = batch_size
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        self.dim = len(self._request(["dimension probe"])[0])

    def _request(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post("/embeddings", json={"model": self.model, "input": texts})
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._request(texts[start : start + self.batch_size]))
        return _normalize_rows(np.array(vectors, dtype=np.float32))


class AzureOpenAIEmbedder(OpenAIEmbedder):
    name = "azure"

    def __init__(self, endpoint: str, api_key: str, deployment: str, api_version: str, batch_size: int = 64):
        if not (endpoint and api_key and deployment):
            raise ValueError(
                "CITADOC_AZURE_OPENAI_ENDPOINT, CITADOC_AZURE_OPENAI_API_KEY and "
                "CITADOC_AZURE_OPENAI_EMBEDDING_DEPLOYMENT are required"
            )
        self.deployment = deployment
        self.api_version = api_version
        self.batch_size = batch_size
        self._client = httpx.Client(
            base_url=endpoint.rstrip("/"), headers={"api-key": api_key}, timeout=60
        )
        self.dim = len(self._request(["dimension probe"])[0])

    def _request(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(
            f"/openai/deployments/{self.deployment}/embeddings",
            params={"api-version": self.api_version},
            json={"input": texts},
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]
