"""Cleaning is deterministic and removes noise without losing meaning."""

from src.ingestion.clean_text import clean_transcript


def test_removes_stage_directions():
    out = clean_transcript("Thank you. (Applause) Let's begin. (Laughter)")
    assert "Applause" not in out and "Laughter" not in out
    assert "Thank you" in out and "Let's begin" in out


def test_removes_timestamps():
    out = clean_transcript("At 00:12 we start, then [01:02:33] we finish.")
    assert "00:12" not in out and "01:02:33" not in out
    assert "we start" in out and "we finish" in out


def test_removes_html_tags():
    out = clean_transcript("This is <b>bold</b> and <i>italic</i> text.")
    assert "<b>" not in out and "</i>" not in out
    assert "bold" in out and "italic" in out


def test_normalizes_whitespace():
    out = clean_transcript("Too    many     spaces\n\n\nand newlines .")
    assert "  " not in out
    assert out == "Too many spaces and newlines."


def test_empty_input_does_not_crash():
    assert clean_transcript("") == ""
    assert clean_transcript("   \n\t ") == ""
    assert clean_transcript(None) == ""


def test_preserves_meaningful_text():
    text = "Leadership is about making others feel safe and valued."
    assert clean_transcript(text) == text


def test_deterministic():
    text = "Some text (Laughter) with 00:30 noise <p>tags</p>."
    assert clean_transcript(text) == clean_transcript(text)
