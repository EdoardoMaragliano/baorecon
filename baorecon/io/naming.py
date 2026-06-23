"""Tokenized output naming helpers."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


class NamingTokenizer:
    """Render tokenized filename patterns."""

    @staticmethod
    def format_name(pattern: str, **tokens: Any) -> str:
        """Format a naming pattern with built-in date/time tokens."""
        values = dict(tokens)
        now = datetime.now()
        values.setdefault("date", now.strftime("%Y-%m-%d"))
        values.setdefault("time", now.strftime("%H-%M-%S"))

        try:
            rendered = pattern.format(**values)
        except KeyError as exc:
            raise ValueError("Unknown token in naming pattern: {0}".format(exc))

        rendered = re.sub(r"[^A-Za-z0-9_\-.]+", "_", rendered)
        logger.debug("Generated output name: {0}".format(rendered))
        return rendered
