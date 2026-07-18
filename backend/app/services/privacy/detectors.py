"""Local, regex-based PII detection.

`PIIDetector` is a minimal, provider-neutral interface (`detect(text) ->
List[PIIEntity]`) so masking orchestration never depends on a concrete
implementation. Two implementations:

- `LocalRegexPIIDetector` (production): pure-Python regex + checksum
  validation (Luhn for cards, ABA checksum for routing numbers). Runs
  entirely on-device -- no cloud PII service, no external API call, no
  model download.
- `FakePIIDetector` (tests): returns a fixed, caller-supplied list of
  entities regardless of input, so masking-pipeline tests can be isolated
  from regex specifics, and can simulate a detector failure.

IMPORTANT LIMITATIONS (see also the README): this is best-effort, regex-
based detection of a fixed set of US-centric, structurally-recognizable
identifiers. It does NOT reliably detect personal names, free-form postal
addresses, or most non-US identifiers, and it is not a substitute for a
professional PII/DLP tool. It makes no HIPAA/PCI-DSS/GDPR/GLBA or other
regulatory compliance claim. See `README.md` before using this with real
customer or other sensitive data -- this portfolio project is intended
for synthetic/public sample documents only.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import List, Optional

from app.core.config import get_settings

# --- Categories & placeholders ----------------------------------------------

CATEGORY_SSN = "SSN"
CATEGORY_CARD = "CARD"
CATEGORY_EMAIL = "EMAIL"
CATEGORY_PHONE = "PHONE"
CATEGORY_IPV4 = "IPV4"
CATEGORY_DOB = "DOB"
CATEGORY_BANK_ACCOUNT = "BANK_ACCOUNT"
CATEGORY_ROUTING_NUMBER = "ROUTING_NUMBER"

_PLACEHOLDERS = {
    CATEGORY_SSN: "[SSN_REDACTED]",
    CATEGORY_CARD: "[CARD_REDACTED]",
    CATEGORY_EMAIL: "[EMAIL_REDACTED]",
    CATEGORY_PHONE: "[PHONE_REDACTED]",
    CATEGORY_IPV4: "[IP_REDACTED]",
    CATEGORY_DOB: "[DOB_REDACTED]",
    CATEGORY_BANK_ACCOUNT: "[ACCOUNT_REDACTED]",
    CATEGORY_ROUTING_NUMBER: "[ROUTING_REDACTED]",
}

# Detector priority when two candidate entities overlap -- lower wins.
# Context-gated, checksum-validated categories are the most specific and
# take precedence over structurally-looser ones (see `_resolve_overlaps`).
_CATEGORY_PRIORITY = {
    CATEGORY_SSN: 0,
    CATEGORY_CARD: 1,
    CATEGORY_ROUTING_NUMBER: 2,
    CATEGORY_BANK_ACCOUNT: 3,
    CATEGORY_DOB: 4,
    CATEGORY_EMAIL: 5,
    CATEGORY_PHONE: 6,
    CATEGORY_IPV4: 7,
}


@dataclass(frozen=True)
class PIIEntity:
    """One detected span of sensitive text. Never carries the matched
    value itself beyond what's needed to mask it in place -- callers must
    not log `category`/`start`/`end` alongside the source text."""

    category: str
    start: int
    end: int
    placeholder: str


class PIIDetector(ABC):
    """Turns raw text into a list of non-overlapping `PIIEntity` spans."""

    @abstractmethod
    def detect(self, text: str) -> List[PIIEntity]:
        """Return entities found in `text`, sorted by `start`, non-overlapping."""


def _resolve_overlaps(entities: List[PIIEntity]) -> List[PIIEntity]:
    """Deterministically resolve overlapping candidate entities.

    Sorted by start position; when a later candidate overlaps an already-
    accepted one, it is kept only if it doesn't intersect. Ties in
    overlap are broken by `_CATEGORY_PRIORITY` (lower wins), then by
    longer span, then by earlier start -- all deterministic, no reliance
    on dict/set iteration order.
    """

    ordered = sorted(
        entities,
        key=lambda e: (e.start, _CATEGORY_PRIORITY.get(e.category, 99), -(e.end - e.start)),
    )

    accepted: List[PIIEntity] = []
    for candidate in ordered:
        conflict_index = None
        for index, existing in enumerate(accepted):
            if candidate.start < existing.end and existing.start < candidate.end:
                conflict_index = index
                break

        if conflict_index is None:
            accepted.append(candidate)
            continue

        existing = accepted[conflict_index]
        candidate_key = (_CATEGORY_PRIORITY.get(candidate.category, 99), -(candidate.end - candidate.start))
        existing_key = (_CATEGORY_PRIORITY.get(existing.category, 99), -(existing.end - existing.start))
        if candidate_key < existing_key:
            accepted[conflict_index] = candidate

    return sorted(accepted, key=lambda e: e.start)


def _luhn_is_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = ord(char) - ord("0")
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _routing_checksum_is_valid(digits: str) -> bool:
    if len(digits) != 9:
        return False
    d = [ord(c) - ord("0") for c in digits]
    checksum = 3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + 1 * (d[2] + d[5] + d[8])
    return checksum % 10 == 0


def _ssn_digits_are_valid(digits: str) -> bool:
    """Reject 9-digit strings that can never be a real SSN.

    Per SSA rules: the area (first 3) is never 000, 666, or 900-999; the
    group (middle 2) is never 00; the serial (last 4) is never 0000. A
    candidate failing these is structurally impossible as an SSN, so
    treating it as one would be a pure false positive.
    """

    if len(digits) != 9:
        return False
    area, group, serial = digits[0:3], digits[3:5], digits[5:9]
    if area == "000" or area == "666" or area[0] == "9":
        return False
    if group == "00":
        return False
    if serial == "0000":
        return False
    return True


_DATE_NUMERIC = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$")
_DATE_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_MONTH_NAME_FORMATS = ("%B %d %Y", "%b %d %Y")


def _is_plausible_calendar_date(value: str) -> bool:
    """Reject date-shaped strings that can never be a real calendar date.

    The numeric `\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}` pattern doesn't encode
    whether it's month/day or day/month order, so a candidate is accepted
    if *either* interpretation is a valid date -- this only ever rejects
    values that are impossible under both readings (e.g. "13/45/2020").
    """

    match = _DATE_NUMERIC.match(value)
    if match:
        first, second, year_str = match.groups()
        year = int(year_str)
        if len(year_str) == 2:
            year += 2000 if year < 70 else 1900
        first, second = int(first), int(second)
        for month, day in ((first, second), (second, first)):
            try:
                date(year, month, day)
                return True
            except ValueError:
                continue
        return False

    match = _DATE_ISO.match(value)
    if match:
        year, month, day = (int(part) for part in match.groups())
        try:
            date(year, month, day)
            return True
        except ValueError:
            return False

    # Handles both "January 15 1985" and an abbreviated-with-dot form like
    # "Jan. 15 2020" (the dot is optional in `_DATE_VALUE`).
    normalized = value.replace(",", "").strip()
    for candidate in {normalized, normalized.replace(".", "")}:
        for fmt in _MONTH_NAME_FORMATS:
            try:
                datetime.strptime(candidate, fmt)
                return True
            except ValueError:
                continue

    return False


class LocalRegexPIIDetector(PIIDetector):
    """Production PII detector: pure regex + checksum validation, on-device only."""

    # ASCII-only \d so surrounding Unicode text can never influence which
    # characters are treated as digits, and matched digit strings always
    # parse predictably for Luhn/checksum validation.
    _SSN_FORMATTED = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)", re.ASCII)
    _SSN_CONTEXT = re.compile(r"\b(?:ssn|social\s+security(?:\s+number)?)\b\W{0,10}", re.IGNORECASE | re.ASCII)
    _NINE_DIGITS = re.compile(r"(?<!\d)\d{9}(?!\d)", re.ASCII)

    _CARD_FORMATTED = re.compile(r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7}(?!\d)", re.ASCII)
    _CARD_UNFORMATTED = re.compile(r"(?<!\d)\d{13,19}(?!\d)", re.ASCII)

    _EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    _PHONE_FORMATTED = re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)", re.ASCII
    )
    _PHONE_UNFORMATTED = re.compile(r"(?<!\d)[2-9]\d{9}(?!\d)", re.ASCII)

    _IPV4 = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b", re.ASCII
    )

    _DOB_CONTEXT = re.compile(
        r"\b(?:date\s+of\s+birth|dob|birth\s*date|born(?:\s+on)?)\b\W{0,10}", re.IGNORECASE | re.ASCII
    )
    _DATE_VALUE = re.compile(
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z]*\.?\s+\d{1,2},?\s+\d{4}",
        re.IGNORECASE | re.ASCII,
    )

    _ROUTING_CONTEXT = re.compile(
        r"\b(?:routing\s*(?:number|no\.?|#)?|aba(?:\s*number)?|rtn)\b\W{0,10}", re.IGNORECASE | re.ASCII
    )
    _ACCOUNT_CONTEXT = re.compile(
        r"\b(?:account\s*(?:number|no\.?|#)|acct\.?\s*(?:number|no\.?|#)?|a/c\s*(?:no\.?|#)?)\b\W{0,10}",
        re.IGNORECASE | re.ASCII,
    )
    _ACCOUNT_NUMBER_VALUE = re.compile(r"(?<!\d)\d{6,17}(?!\d)", re.ASCII)

    _CONTEXT_WINDOW = 25

    def __init__(
        self,
        mask_emails: bool = True,
        mask_phones: bool = True,
        mask_financial_identifiers: bool = True,
    ) -> None:
        self._mask_emails = mask_emails
        self._mask_phones = mask_phones
        self._mask_financial_identifiers = mask_financial_identifiers

    def detect(self, text: str) -> List[PIIEntity]:
        candidates: List[PIIEntity] = []

        candidates += self._find_ssn(text)
        candidates += self._find_dob(text)

        if self._mask_financial_identifiers:
            candidates += self._find_cards(text)
            candidates += self._find_routing_numbers(text)
            candidates += self._find_bank_accounts(text)

        if self._mask_emails:
            candidates += self._find_emails(text)

        if self._mask_phones:
            candidates += self._find_phones(text)

        candidates += self._find_ipv4(text)

        return _resolve_overlaps(candidates)

    # --- Individual category finders -----------------------------------

    def _find_ssn(self, text: str) -> List[PIIEntity]:
        entities = [
            self._entity(CATEGORY_SSN, match.start(), match.end())
            for match in self._SSN_FORMATTED.finditer(text)
            if _ssn_digits_are_valid(re.sub(r"\D", "", match.group()))
        ]
        entities += [
            entity
            for entity in self._find_context_gated_numbers(text, self._SSN_CONTEXT, self._NINE_DIGITS, CATEGORY_SSN)
            if _ssn_digits_are_valid(text[entity.start : entity.end])
        ]
        return entities

    def _find_cards(self, text: str) -> List[PIIEntity]:
        entities = []
        for pattern in (self._CARD_FORMATTED, self._CARD_UNFORMATTED):
            for match in pattern.finditer(text):
                digits = re.sub(r"\D", "", match.group())
                if 13 <= len(digits) <= 19 and _luhn_is_valid(digits):
                    entities.append(self._entity(CATEGORY_CARD, match.start(), match.end()))
        return entities

    def _find_emails(self, text: str) -> List[PIIEntity]:
        return [self._entity(CATEGORY_EMAIL, match.start(), match.end()) for match in self._EMAIL.finditer(text)]

    def _find_phones(self, text: str) -> List[PIIEntity]:
        entities = [
            self._entity(CATEGORY_PHONE, match.start(), match.end())
            for match in self._PHONE_FORMATTED.finditer(text)
        ]
        entities += [
            self._entity(CATEGORY_PHONE, match.start(), match.end())
            for match in self._PHONE_UNFORMATTED.finditer(text)
        ]
        return entities

    def _find_ipv4(self, text: str) -> List[PIIEntity]:
        return [self._entity(CATEGORY_IPV4, match.start(), match.end()) for match in self._IPV4.finditer(text)]

    def _find_dob(self, text: str) -> List[PIIEntity]:
        entities = []
        for context_match in self._DOB_CONTEXT.finditer(text):
            window_start = context_match.end()
            window = text[window_start : window_start + self._CONTEXT_WINDOW]
            date_match = self._DATE_VALUE.search(window)
            if date_match and _is_plausible_calendar_date(date_match.group()):
                entities.append(
                    self._entity(
                        CATEGORY_DOB,
                        window_start + date_match.start(),
                        window_start + date_match.end(),
                    )
                )
        return entities

    def _find_routing_numbers(self, text: str) -> List[PIIEntity]:
        entities = []
        for context_match in self._ROUTING_CONTEXT.finditer(text):
            window_start = context_match.end()
            window = text[window_start : window_start + self._CONTEXT_WINDOW]
            number_match = self._NINE_DIGITS.search(window)
            if number_match and _routing_checksum_is_valid(number_match.group()):
                entities.append(
                    self._entity(
                        CATEGORY_ROUTING_NUMBER,
                        window_start + number_match.start(),
                        window_start + number_match.end(),
                    )
                )
        return entities

    def _find_bank_accounts(self, text: str) -> List[PIIEntity]:
        entities = []
        for context_match in self._ACCOUNT_CONTEXT.finditer(text):
            window_start = context_match.end()
            window = text[window_start : window_start + self._CONTEXT_WINDOW]
            number_match = self._ACCOUNT_NUMBER_VALUE.search(window)
            if number_match:
                entities.append(
                    self._entity(
                        CATEGORY_BANK_ACCOUNT,
                        window_start + number_match.start(),
                        window_start + number_match.end(),
                    )
                )
        return entities

    def _find_context_gated_numbers(
        self, text: str, context_pattern: "re.Pattern", value_pattern: "re.Pattern", category: str
    ) -> List[PIIEntity]:
        entities = []
        for context_match in context_pattern.finditer(text):
            window_start = context_match.end()
            window = text[window_start : window_start + self._CONTEXT_WINDOW]
            value_match = value_pattern.search(window)
            if value_match:
                entities.append(
                    self._entity(category, window_start + value_match.start(), window_start + value_match.end())
                )
        return entities

    @staticmethod
    def _entity(category: str, start: int, end: int) -> PIIEntity:
        return PIIEntity(category=category, start=start, end=end, placeholder=_PLACEHOLDERS[category])


class FakePIIDetector(PIIDetector):
    """Deterministic, fully-controlled detector for tests.

    Returns a fixed, caller-supplied list of entities regardless of
    input text (so masking-pipeline tests are isolated from regex
    specifics), or raises a caller-supplied exception to exercise error
    handling.
    """

    def __init__(
        self,
        entities: Optional[List[PIIEntity]] = None,
        raise_exception: Optional[BaseException] = None,
    ) -> None:
        self._entities = entities or []
        self._raise_exception = raise_exception

    def detect(self, text: str) -> List[PIIEntity]:
        if self._raise_exception is not None:
            raise self._raise_exception
        return list(self._entities)


@lru_cache
def get_pii_detector() -> PIIDetector:
    """Process-wide singleton `PIIDetector`, wired from Settings.

    Tests override this dependency (via `app.dependency_overrides`) with
    a `FakePIIDetector` when they need fully controlled entities;
    `LocalRegexPIIDetector` itself has no network/model dependency, so
    using the real one directly in tests is also safe.
    """

    settings = get_settings()
    return LocalRegexPIIDetector(
        mask_emails=settings.pii_mask_emails,
        mask_phones=settings.pii_mask_phones,
        mask_financial_identifiers=settings.pii_mask_financial_identifiers,
    )
