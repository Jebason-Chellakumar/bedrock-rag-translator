"""Placeholders, glossary, QA and document segmentation."""

import pytest

from translator import qa
from translator.glossary import Glossary, Term
from translator.placeholders import mask, token_problems, unmask
from translator.segments import join_document, split_document


@pytest.mark.parametrize(
    "text, originals",
    [
        ("Log in at {{company_url}} now", ["{{company_url}}"]),
        ("Hi {name}, you have %(count)d items and %s", ["{name}", "%(count)d", "%s"]),
        ('Click <a href="{{renew_url}}">here</a>.', ['<a href="{{renew_url}}">', "</a>"]),
        ("See https://help.example.com/a?b=1.", ["https://help.example.com/a?b=1"]),
        ("Nothing to protect here.", []),
    ],
)
def test_mask_and_unmask_round_trip(text, originals):
    m = mask(text)
    assert list(m.originals.values()) == originals
    for original in originals:
        assert original not in m.text
    assert unmask(m.text, m) == text


def test_token_problems():
    m = mask("{a} and {b}")
    assert token_problems("⟦P1⟧ und ⟦P0⟧", m) == []  # reordering is fine
    assert token_problems("⟦P0⟧ und", m) == ["placeholder {b} is missing"]
    assert token_problems("⟦P0⟧ ⟦P0⟧ ⟦P1⟧", m) == ["placeholder {a} appears 2 times"]
    assert token_problems("⟦P0⟧ ⟦P1⟧ ⟦P7⟧", m) == ["unknown placeholder token ⟦P7⟧"]


def test_glossary_matching(glossary):
    hits = lambda text: [t.term for t in glossary.matches(text, "de-DE")]  # noqa: E731
    assert hits("You have 3 courses due") == ["course"]  # plural
    assert hits("Your course {course} is ready") == ["course"]
    assert hits("Details for {course}") == []  # placeholder, not the word
    assert hits("Visit https://help.learnhub.example") == []  # URL, not the product name
    assert hits("Open LearnHub to continue your learning path") == ["learning path", "LearnHub"]
    assert [t.term for t in glossary.matches("course", "it-IT")] == []  # no Italian entry


def test_glossary_violations(glossary):
    assert glossary.violations("Your course", "Ihr Kurs", "de-DE") == []
    assert glossary.violations("Your courses", "Ihre Kurse", "de-DE") == []
    assert glossary.violations("Your course", "Ihr Lehrgang", "de-DE") == [
        "glossary term 'course' must be rendered as 'Kurs'"
    ]
    assert glossary.violations("Open LearnHub", "Öffnen Sie LernHub", "de-DE") == [
        "do-not-translate term 'LearnHub' must be rendered as 'LearnHub'"
    ]


def check(source, translated_masked, locale="de-DE", glossary=None):
    m = mask(source)
    return qa.check(source, m, translated_masked, unmask(translated_masked, m), locale, glossary or Glossary([]))


def test_qa_errors_and_warnings():
    assert check("Hello {name}", "Hallo ⟦P0⟧") == []
    assert qa.has_errors(check("Hello {name}", "Hallo"))
    assert qa.has_errors(check("Hello", "   "))
    assert [str(i) for i in check("Pass with 80% or more", "Mit mindestens 70 % bestehen")] == [
        "warning: number 80 not found in translation"
    ]
    assert check("Costs 1.50 today", "Kostet heute 1,50") == []  # decimal separator varies
    same = check("Please complete your training today", "Please complete your training today")
    assert [i.message for i in same] == ["translation is identical to the source"]
    long = check("Please complete it today now", "Bitte " * 30)
    assert any("length ratio" in i.message for i in long)
    assert check("Please complete your training today", "今日中に研修を完了してください", "ja-JP") == []


def test_qa_glossary_error():
    g = Glossary([Term("course", "de-DE", "Kurs")])
    assert [i.message for i in check("Start the course", "Starten Sie die Schulung", glossary=g)] == [
        "glossary term 'course' must be rendered as 'Kurs'"
    ]


def test_document_split_and_join_skips_code():
    doc = "# Title\n\nFirst paragraph.\n\n```bash\nmake build\n\nmake test\n```\n\nLast paragraph.\n"
    segments = split_document(doc)
    assert [(s.id, s.text, s.context) for s in segments] == [
        ("p1", "# Title", "heading"),
        ("p2", "First paragraph.", "paragraph"),
        ("p5", "Last paragraph.", "paragraph"),
    ]
    rebuilt = join_document(doc, {"p1": "# Titel", "p2": "Erster Absatz.", "p5": "Letzter Absatz."})
    assert rebuilt == "# Titel\n\nErster Absatz.\n\n```bash\nmake build\n\nmake test\n```\n\nLetzter Absatz.\n"
