"""Tests for baorecon/io/naming.py — NamingTokenizer."""

from __future__ import annotations

import re
from unittest.mock import patch
from datetime import datetime

import pytest

from baorecon.io.naming import NamingTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_NOW = datetime(2026, 5, 21, 18, 30, 0)
FIXED_DATE = "2026-05-21"
FIXED_TIME = "18-30-00"


@pytest.fixture(autouse=True)
def freeze_time():
    """Patch datetime.now() to a fixed value in all tests."""
    with patch("baorecon.io.naming.datetime") as mock_dt:
        mock_dt.now.return_value = FIXED_NOW
        mock_dt.strftime = datetime.strftime  # delega i metodi di istanza
        yield mock_dt


# ---------------------------------------------------------------------------
# Token substitution
# ---------------------------------------------------------------------------

class TestTokenSubstitution:

    def test_single_token(self):
        result = NamingTokenizer.format_name("catalog_{bias}", bias=1.5)
        assert "1.5" in result

    def test_multiple_tokens(self):
        result = NamingTokenizer.format_name(
            "recon_b{bias}_s{smoothing}_N{ngrid}",
            bias=1.5, smoothing=15, ngrid=512
        )
        assert "b1.5" in result
        assert "s15" in result
        assert "N512" in result

    def test_unknown_token_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown token"):
            NamingTokenizer.format_name("{missing_token}", other=1)

    def test_empty_pattern(self):
        result = NamingTokenizer.format_name("")
        assert isinstance(result, str)

    def test_pattern_no_tokens(self):
        result = NamingTokenizer.format_name("fixed_output")
        assert result == "fixed_output"


# ---------------------------------------------------------------------------
# Auto date/time injection
# ---------------------------------------------------------------------------

class TestAutoDateTimeTokens:

    def test_date_injected_automatically(self):
        result = NamingTokenizer.format_name("run_{date}")
        assert FIXED_DATE in result

    def test_time_injected_automatically(self):
        result = NamingTokenizer.format_name("run_{time}")
        assert FIXED_TIME in result

    def test_user_can_override_date(self):
        """User-supplied date must take precedence over auto-generated one."""
        result = NamingTokenizer.format_name("run_{date}", date="1970-01-01")
        assert "1970-01-01" in result
        assert FIXED_DATE not in result

    def test_user_can_override_time(self):
        result = NamingTokenizer.format_name("run_{time}", time="00-00-00")
        assert "00-00-00" in result


# ---------------------------------------------------------------------------
# Sanitization (regex sub)
# ---------------------------------------------------------------------------

class TestSanitization:

    def test_spaces_replaced(self):
        result = NamingTokenizer.format_name("output {bias}", bias=1.5)
        assert " " not in result

    def test_slash_replaced(self):
        result = NamingTokenizer.format_name("dir/file_{bias}", bias=1.5)
        assert "/" not in result

    def test_only_allowed_chars_remain(self):
        """Output must only contain A-Za-z0-9, _, -, ."""
        result = NamingTokenizer.format_name(
            "recon!@#b{bias}$$s{smoothing}", bias=1.5, smoothing=15
        )
        assert re.fullmatch(r"[A-Za-z0-9_\-.]+", result), \
            f"Illegal characters in: {result!r}"

    def test_allowed_chars_preserved(self):
        """Dots, dashes, underscores devono restare intatti."""
        result = NamingTokenizer.format_name("recon_b{bias}-v1.0", bias=2.0)
        assert "recon_b2.0-v1.0" == result


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_inputs_same_output(self):
        r1 = NamingTokenizer.format_name("{bias}_{smoothing}", bias=1.5, smoothing=15)
        r2 = NamingTokenizer.format_name("{bias}_{smoothing}", bias=1.5, smoothing=15)
        assert r1 == r2

    def test_different_bias_different_output(self):
        r1 = NamingTokenizer.format_name("{bias}", bias=1.5)
        r2 = NamingTokenizer.format_name("{bias}", bias=2.0)
        assert r1 != r2

    def test_different_smoothing_different_output(self):
        r1 = NamingTokenizer.format_name("{smoothing}", smoothing=10)
        r2 = NamingTokenizer.format_name("{smoothing}", smoothing=20)
        assert r1 != r2