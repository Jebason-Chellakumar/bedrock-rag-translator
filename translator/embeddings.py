"""Sentence embeddings: Amazon Titan Text Embeddings V2 on Bedrock, or an offline hasher."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import numpy as np

TITAN_V2 = "amazon.titan-embed-text-v2:0"


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dimensions) float32 array of L2-normalised vectors."""


class TitanEmbedder:
    """Titan Text Embeddings V2 via ``bedrock-runtime`` InvokeModel.

    Titan embeds one text per request, so requests run concurrently.
    """

    def __init__(self, client: Any = None, region: str | None = None, dimensions: int = 512):
        if dimensions not in (256, 512, 1024):
            raise ValueError("Titan V2 supports 256, 512 or 1024 dimensions")
        if client is None:
            import boto3

            client = boto3.client("bedrock-runtime", region_name=region)
        self.client = client
        self.dimensions = dimensions

    def _one(self, text: str) -> list:
        body = json.dumps({"inputText": text, "dimensions": self.dimensions, "normalize": True})
        resp = self.client.invoke_model(
            modelId=TITAN_V2, body=body, contentType="application/json", accept="application/json"
        )
        return json.loads(resp["body"].read())["embedding"]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=8) as pool:
            vectors = list(pool.map(self._one, texts))
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Deterministic, offline lexical embedding (word + character trigram hashing).

    Not semantic like Titan, but good enough to demo retrieval and to test the
    pipeline without AWS credentials.
    """

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions

    def _features(self, text: str):
        text = text.lower()
        words = re.findall(r"\w+", text)
        yield from ((f"w:{w}", 1.0) for w in words)
        padded = f" {' '.join(words)} "
        yield from ((f"c:{padded[i : i + 3]}", 0.5) for i in range(len(padded) - 2))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dimensions] += weight if (h >> 63) & 1 else -weight
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out
