import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import clean_text


def test_clean_text_removes_html_tags():
    raw = "This movie was great!<br /><br />Would watch again."
    cleaned = clean_text(raw)
    assert "<br" not in cleaned
    assert "great" in cleaned
    assert "Would watch again" in cleaned


def test_clean_text_collapses_whitespace():
    raw = "Too   many      spaces   here"
    cleaned = clean_text(raw)
    assert "  " not in cleaned


def test_clean_text_strips_leading_trailing_whitespace():
    raw = "   padded text   "
    cleaned = clean_text(raw)
    assert cleaned == "padded text"


def test_clean_text_handles_empty_string():
    assert clean_text("") == ""
