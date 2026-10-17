import json
from types import SimpleNamespace as NS

import pytest
from conftest import SAMPLE, FakeTranslator

from translator.glossary import Glossary
from translator.llm import (
    SCHEMA,
    BatchItem,
    BatchTooLarge,
    ClaudeTranslator,
    TranslationRefused,
    build_prompt,
)
from translator.pipeline import NEEDS_REVIEW, TM_EXACT, TRANSLATED, Pipeline
from translator.segments import Segment, read_strings

SEGMENTS = read_strings(SAMPLE / "strings.csv")


def by_id(rows):
    return {r.id: r for r in rows}


def test_memory_hits_skip_the_model_and_the_rest_pass_qa(memory, glossary):
    fake = FakeTranslator()
    results = Pipeline(fake, memory, glossary).run(SEGMENTS, ["de-DE", "ja-JP"])
    de = by_id(results["de-DE"])
    assert [r.id for r in results["de-DE"]] == [s.id for s in SEGMENTS]  # input order kept
    assert {k for k, r in de.items() if r.status == TM_EXACT} == {"login.reminder", "cert.ready", "auth.logout"}
    assert de["auth.logout"].target == "Abmelden"
    sent = {i for locale, ids in fake.calls if locale == "de-DE" for i in ids}
    assert not sent & {"login.reminder", "cert.ready", "auth.logout"}
    assert all(r.status == TRANSLATED for r in de.values() if r.status != TM_EXACT)
    # placeholders restored exactly
    assert de["cert.expiring"].target.count('<a href="{{renew_url}}">') == 1
    assert "{course}" in de["supervisor.assigned"].target
    assert de["course.due_today"].references >= 1


def test_qa_failure_is_retried_with_feedback(memory, glossary):
    fake = FakeTranslator(break_tokens={"course.overdue"})
    [result] = [
        r for r in Pipeline(fake, memory, glossary).run(SEGMENTS, ["fr-FR"])["fr-FR"] if r.id == "course.overdue"
    ]
    assert result.status == TRANSLATED and result.attempts == 2
    assert fake.calls[-1] == ("fr-FR", ["course.overdue"])  # only the failing segment is resent


def test_persistent_failure_needs_review(memory, glossary):
    fake = FakeTranslator(always_break={"course.overdue"})
    result = by_id(Pipeline(fake, memory, glossary).run(SEGMENTS, ["es-MX"])["es-MX"])["course.overdue"]
    assert result.status == NEEDS_REVIEW and result.attempts == 2
    assert "placeholder {name} is missing" in [i.message for i in result.issues]


def test_large_batches_are_split(memory, glossary):
    fake = FakeTranslator(max_batch=2)
    results = Pipeline(fake, memory, glossary, batch_size=8).run(SEGMENTS, ["de-DE"])["de-DE"]
    assert all(r.status != NEEDS_REVIEW for r in results)
    assert max(len(ids) for _, ids in fake.calls if len(ids) <= 2) <= 2


def test_refusal_marks_batch_for_review(memory, glossary):
    fake = FakeTranslator(refuse={"ja-JP"})
    results = Pipeline(fake, memory, glossary).run(SEGMENTS, ["de-DE", "ja-JP"])
    ja = by_id(results["ja-JP"])
    assert ja["cert.ready"].status == TM_EXACT  # memory still works
    assert ja["path.continue"].status == NEEDS_REVIEW
    assert "model declined" in ja["path.continue"].issues[0].message
    assert all(r.status != NEEDS_REVIEW for r in results["de-DE"])


def test_build_prompt_contains_context():
    items = [
        BatchItem("a", "Start ⟦P0⟧", "button", (("Start path", "Pfad starten", 0.7),), (("course", "Kurs", False),)),
        BatchItem(
            "b",
            "Open LearnHub",
            "",
            (("Start path", "Pfad starten", 0.9),),
            (("LearnHub", "LearnHub", True),),
            "fix it",
        ),
    ]
    prompt = build_prompt("de-DE", items)
    assert "<target_locale>de-DE</target_locale>" in prompt and "Formal address (Sie)" in prompt
    assert "- course => Kurs" in prompt and "- LearnHub => LearnHub (do not translate)" in prompt
    assert prompt.count("Pfad starten") == 1  # references are de-duplicated
    segments = json.loads(prompt.split("<segments>")[1].split("</segments>")[0])
    assert segments[0] == {"id": "a", "text": "Start ⟦P0⟧", "context": "button"}
    assert segments[1]["qa_feedback"] == "fix it"
    assert "Natural, professional register" in build_prompt("xx-XX", items[:1])


class FakeClaude:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def claude_response(payload=None, stop="end_turn"):
    content = [NS(type="thinking", thinking="")]
    if payload is not None:
        content.append(NS(type="text", text=json.dumps(payload)))
    usage = NS(input_tokens=1200, output_tokens=300, cache_read_input_tokens=900)
    return NS(content=content, stop_reason=stop, usage=usage)


def test_claude_translator_request_and_parsing():
    client = FakeClaude(claude_response({"translations": [{"id": "a", "translation": "Hallo"}]}))
    t = ClaudeTranslator(client=client)
    assert t.translate("de-DE", [BatchItem("a", "Hello")]) == {"a": "Hallo"}
    req = client.requests[0]
    assert req["model"] == "anthropic.claude-opus-5-5"
    assert req["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert req["cache_control"] == {"type": "ephemeral"}
    assert "⟦P0⟧" in req["system"]  # stable system prompt, cacheable
    assert t.usage.requests == 1 and t.usage.cache_read_tokens == 900


@pytest.mark.parametrize("stop, error", [("refusal", TranslationRefused), ("max_tokens", BatchTooLarge)])
def test_claude_translator_stop_reasons(stop, error):
    t = ClaudeTranslator(client=FakeClaude(claude_response(stop=stop)))
    with pytest.raises(error):
        t.translate("de-DE", [BatchItem("a", "Hello")])


def test_plan_for_dry_run(memory, glossary):
    plans = Pipeline(None, memory, glossary).plan([Segment("x", "Continue learning path")], "de-DE")
    assert plans[0].exact is None
    assert [t.term for t in plans[0].terms] == ["learning path"]
    assert plans[0].matches[0].entry.target == "Lernpfad starten"


def test_empty_glossary_and_memory_still_translate():
    from translator.embeddings import HashingEmbedder
    from translator.memory import TranslationMemory

    results = Pipeline(FakeTranslator(), TranslationMemory(HashingEmbedder()), Glossary([])).run(
        [Segment("a", "Hello {name}")], ["it-IT"]
    )
    assert results["it-IT"][0].status == TRANSLATED and "{name}" in results["it-IT"][0].target
