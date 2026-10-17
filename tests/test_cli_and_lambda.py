import csv
import shutil

from conftest import SAMPLE, FakeTranslator

from translator import lambda_handler
from translator.cli import build_parser, main


def run_cli(argv, translator=None):
    args = build_parser().parse_args(argv)
    return args.func(args, translator) if translator else args.func(args)


def common(tmp_path):
    return ["--glossary", str(SAMPLE / "glossary.csv"), "--memory", str(SAMPLE / "memory.csv"), "--embedder", "hash"]


def test_dry_run_needs_no_credentials(tmp_path, capsys):
    code = main(["strings", str(SAMPLE / "strings.csv"), "--locales", "de-DE", "--dry-run", *common(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "memory: exact match, reused without a model call -> Abmelden" in out
    assert "glossary: learning path => Lernpfad" in out


def test_strings_command_writes_table_report_and_updates_memory(tmp_path):
    memory = tmp_path / "memory.csv"
    shutil.copy(SAMPLE / "memory.csv", memory)
    out, report = tmp_path / "out.csv", tmp_path / "report.md"
    argv = [
        "strings", str(SAMPLE / "strings.csv"), "--locales", "de-DE,fr-FR",
        "--out", str(out), "--report", str(report), "--update-memory",
        "--glossary", str(SAMPLE / "glossary.csv"), "--memory", str(memory), "--embedder", "hash",
    ]  # fmt: skip
    assert run_cli(argv, FakeTranslator()) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert list(rows[0]) == ["key", "source", "de-DE", "fr-FR"]
    assert {r["key"]: r["de-DE"] for r in rows}["auth.logout"] == "Abmelden"
    text = report.read_text()
    assert "| de-DE | 12 | 3 | 9 | 0 | 0 |" in text and "Every segment passed QA" in text
    with open(memory, encoding="utf-8") as fh:
        assert sum(1 for _ in fh) - 1 == 20 + 18  # 2 locales x 9 newly translated segments


def test_strings_command_exit_code_signals_review(tmp_path):
    argv = [
        "strings",
        str(SAMPLE / "strings.csv"),
        "--locales",
        "de-DE",
        "--out",
        str(tmp_path / "o.csv"),
        *common(tmp_path),
    ]
    assert run_cli(argv, FakeTranslator(always_break={"course.overdue"})) == 1


def test_document_command(tmp_path):
    doc = tmp_path / "guide.md"
    doc.write_text("# Getting started\n\nOpen the course from your learning path.\n", encoding="utf-8")
    out = tmp_path / "guide.de.md"
    argv = ["document", str(doc), "--locale", "de-DE", "--out", str(out), *common(tmp_path)]
    assert run_cli(argv, FakeTranslator()) == 0
    assert out.read_text(encoding="utf-8").startswith("«de-DE» # Getting started")


class FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)

    def download_file(self, bucket, key, path):
        if key not in self.objects:
            err = Exception("missing")
            err.response = {"Error": {"Code": "404"}}
            raise err
        shutil.copy(self.objects[key], path)

    def upload_file(self, path, bucket, key):
        self.objects[key] = open(path, encoding="utf-8").read()


def test_lambda_translates_uploaded_table(monkeypatch):
    monkeypatch.setenv("LOCALES", "de-DE,ja-JP")
    monkeypatch.setenv("EMBEDDER", "hash")
    s3 = FakeS3({"incoming/release+42.csv": SAMPLE / "strings.csv", "config/glossary.csv": SAMPLE / "glossary.csv"})
    event = {
        "Records": [
            {"s3": {"bucket": {"name": "b"}, "object": {"key": "incoming/release%2B42.csv"}}},
            {"s3": {"bucket": {"name": "b"}, "object": {"key": "translated/ignore.csv"}}},
        ]
    }
    result = lambda_handler.handler(event, s3=s3, translator_factory=FakeTranslator)
    assert result == {
        "processed": [
            {"key": "incoming/release+42.csv", "segments": 12, "locales": ["de-DE", "ja-JP"], "needs_review": 0}
        ]
    }
    assert s3.objects["translated/release+42.csv"].startswith("key,source,de-DE,ja-JP")
    assert "# Translation QA report" in s3.objects["translated/release+42.report.md"]
