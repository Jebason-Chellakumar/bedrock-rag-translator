import io
import json

import numpy as np
import pytest

from translator.embeddings import TITAN_V2, HashingEmbedder, TitanEmbedder
from translator.memory import Entry, TranslationMemory, VectorIndex


def test_exact_lookup_ignores_whitespace(memory):
    hit = memory.exact("  Log   out ", "fr-FR")
    assert hit is not None and hit.target == "Se déconnecter"
    assert memory.exact("Log out", "it-IT") is None


def test_search_finds_similar_sentence_in_the_right_locale(memory):
    [matches] = memory.search(["You have {count} courses due today."], "ja-JP")
    assert matches[0].entry.source == "You have {count} courses due this week."
    assert matches[0].entry.locale == "ja-JP"
    assert 0.5 < matches[0].score < 1.0


def test_search_unknown_locale_and_threshold(memory):
    assert memory.search(["Log out"], "it-IT") == [[]]
    [matches] = memory.search(["Quarterly revenue forecast spreadsheet"], "de-DE")
    assert matches == []  # nothing similar enough


def test_faiss_and_numpy_indexes_agree():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 16)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    a, b = VectorIndex(16, use_faiss=True), VectorIndex(16, use_faiss=False)
    a.add(data)
    b.add(data)
    sa, ia = a.search(data[:5], 3)
    sb, ib = b.search(data[:5], 3)
    assert (ia == ib).all() and np.allclose(sa, sb, atol=1e-5)
    assert len(a) == len(b) == 50


def test_save_load_round_trip(tmp_path, memory):
    memory.add([Entry("Continue learning path", "de-DE", "Lernpfad fortsetzen")])
    path = tmp_path / "tm.csv"
    memory.save(path)
    again = TranslationMemory.load(path, HashingEmbedder())
    assert len(again) == len(memory)
    assert again.exact("Continue learning path", "de-DE").target == "Lernpfad fortsetzen"


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(64)
    v = e.embed(["Hello world", "Hello world", ""])
    assert np.allclose(np.linalg.norm(v[:2], axis=1), 1.0)
    assert np.array_equal(v[0], v[1]) and not v[2].any()


class FakeBedrockRuntime:
    def __init__(self):
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        n = len(json.loads(kwargs["body"])["inputText"])
        return {"body": io.BytesIO(json.dumps({"embedding": [float(n)] * 256}).encode())}


def test_titan_embedder_request_shape():
    client = FakeBedrockRuntime()
    vectors = TitanEmbedder(client=client, dimensions=256).embed(["ab", "abcd"])
    assert vectors.shape == (2, 256) and vectors.dtype == np.float32
    assert vectors[1, 0] == 4.0  # results stay in input order despite concurrency
    body = json.loads(client.calls[0]["body"])
    assert client.calls[0]["modelId"] == TITAN_V2
    assert body["dimensions"] == 256 and body["normalize"] is True
    with pytest.raises(ValueError):
        TitanEmbedder(client=client, dimensions=300)
