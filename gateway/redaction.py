"""Personal data and secrets are replaced with placeholders before a prompt leaves the gateway, and
put back into the answer on the way out.

    "Email jane@acme.com about the invoice"  ->  provider sees "Email [EMAIL_1] about the invoice"
    provider answers "Sure, I'll draft it for [EMAIL_1]"  ->  client sees "... for jane@acme.com"

The same value gets the same placeholder throughout a request, so the model can still tell two
people apart. Only the mapping for one request is kept, in memory, while it runs; nothing is stored.

Detection is pattern-based: it catches well-formed emails, card numbers (checked with the Luhn
algorithm), US Social Security numbers, phone numbers written with separators, IPv4 addresses and
common API key formats. It doesn't find names or addresses, which needs a trained model, and it
doesn't scan what the model writes on its own.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from gateway.schemas import ChatRequest

# Most specific first: a card number must not be half-eaten by the phone pattern.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SECRET", re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,}|xox[abprs]-[A-Za-z0-9-]{10,}"
        r"|AIza[0-9A-Za-z_-]{35}|gw_[A-Za-z0-9_-]{40,})")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")),
    ("CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    ("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    ("PHONE", re.compile(   # (555) 123-4567 or 555-123-4567, optionally +1; or international, +44 20 7946 0958
        r"(?<![\w+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}\b"
        r"|(?<![\w+])\+\d{1,3}(?:[ .-]?\d{2,4}){3,5}\b")),
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
]
PLACEHOLDER = re.compile(r"\[(?:SECRET|EMAIL|CARD|SSN|PHONE|IP)_\d+\]")
_MAX_PLACEHOLDER = len("[SECRET_99999]")


def luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


@dataclass
class Vault:
    """One request's placeholders: which value each stands for."""

    originals: dict[str, str] = field(default_factory=dict)     # placeholder -> original value
    _by_value: dict[str, str] = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)             # kind -> distinct values replaced

    def placeholder(self, kind: str, value: str) -> str:
        if value not in self._by_value:
            self.counts[kind] += 1
            token = f"[{kind}_{self.counts[kind]}]"
            self._by_value[value] = token
            self.originals[token] = value
        return self._by_value[value]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def restore(self, text: str) -> str:
        if not self.originals:
            return text
        return PLACEHOLDER.sub(lambda m: self.originals.get(m.group(0), m.group(0)), text)


def redact_text(text: str, vault: Vault) -> str:
    for kind, pattern in PATTERNS:
        def swap(m: re.Match, kind=kind) -> str:
            if kind == "CARD" and not luhn_ok(m.group(0)):
                return m.group(0)
            return vault.placeholder(kind, m.group(0))
        text = pattern.sub(swap, text)
    return text


def redact(chat: ChatRequest) -> tuple[ChatRequest, Vault]:
    vault = Vault()
    messages = [m.model_copy(update={"text": redact_text(m.text, vault)}) for m in chat.messages]
    return chat.model_copy(update={"messages": messages}), vault


class StreamRestorer:
    """Puts originals back into a streamed answer. A placeholder can arrive split across chunks
    ("[EMA" then "IL_1]"), so text that could be the start of one is held back until it either
    completes or clearly isn't one."""

    _maybe_start = re.compile(r"\[[A-Z]*(?:_\d*)?$")

    def __init__(self, vault: Vault):
        self.vault = vault
        self._held = ""

    def feed(self, text: str) -> str:
        text = self._held + text
        self._held = ""
        cut = text.rfind("[")
        if cut != -1 and len(text) - cut < _MAX_PLACEHOLDER and self._maybe_start.match(text, cut):
            text, self._held = text[:cut], text[cut:]
        return self.vault.restore(text)

    def flush(self) -> str:
        text, self._held = self._held, ""
        return self.vault.restore(text)
