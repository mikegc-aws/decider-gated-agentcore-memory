"""Rendering AgentCore memory records into prose.

Not cosmetic. AgentCore's strategies do not all return the same shape: the
`SEMANTIC` strategy hands back plain sentences ("The user is vegetarian.") while
`USER_PREFERENCE` hands back a JSON blob:

    {"context": "Ordering a drink or beverage",
     "preference": "Dislikes sparkling water; prefers still water only",
     "categories": ["beverages", "water"]}

Upstream injects that verbatim into the prompt. That is survivable for the big
model, but it measurably hurts the validator: in a demo run the raw-JSON records
scored systematically higher than prose ones regardless of relevance, so a single
threshold tuned on prose let JSON false positives through ("always books an aisle
seat", kept for a question about what to drink). Flattening both strategies to one
prose style puts them on the same scale.

It also makes the injected `<user_context>` block cheaper and easier to read.
"""

from __future__ import annotations

import json
from typing import Any


def render_memory_text(text: str) -> str:
    """Normalise one memory record's text to a plain sentence.

    Passes prose through untouched; flattens the known JSON shapes. Anything
    unrecognised is returned as-is rather than mangled -- a record we cannot parse
    is still better in the prompt than nothing.
    """
    stripped = text.strip()
    if not stripped.startswith("{") and not stripped.startswith("["):
        return stripped

    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return stripped

    if isinstance(data, list):
        parts = [render_memory_text(json.dumps(d) if not isinstance(d, str) else d) for d in data]
        return "; ".join(p for p in parts if p)

    if not isinstance(data, dict):
        return stripped

    return _render_dict(data) or stripped


def _render_dict(data: dict[str, Any]) -> str:
    """Render the USER_PREFERENCE shape, falling back to key: value pairs."""
    preference = _first_str(data, "preference", "fact", "summary", "content", "text")
    context = _first_str(data, "context", "situation", "topic")

    if preference:
        # "categories" is retrieval metadata, not something the agent needs read out.
        if context:
            return f"When {_lower_first(context)}: {preference}"
        return preference

    # Unknown dict shape: flatten the string-ish values rather than dumping JSON.
    pairs = [
        f"{key}: {value}"
        for key, value in data.items()
        if key != "categories" and isinstance(value, (str, int, float, bool)) and str(value).strip()
    ]
    return "; ".join(pairs)


def _first_str(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _lower_first(text: str) -> str:
    """Lower the first character, unless the first word is an acronym.

    "Ordering a coffee drink" -> "ordering a coffee drink", but "YVR departures"
    must stay "YVR departures".
    """
    if not text:
        return text
    first_word = text.split(maxsplit=1)[0]
    if len(first_word) > 1 and first_word.isupper():
        return text
    return text[0].lower() + text[1:]
