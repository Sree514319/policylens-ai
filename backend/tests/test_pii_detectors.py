"""Tests for local, regex-based PII detection (`app.services.privacy.detectors`).

Pure Python -- no network, no model, no external service.
"""

import pytest

from app.services.privacy.detectors import (
    CATEGORY_BANK_ACCOUNT,
    CATEGORY_CARD,
    CATEGORY_DOB,
    CATEGORY_EMAIL,
    CATEGORY_IPV4,
    CATEGORY_PHONE,
    CATEGORY_ROUTING_NUMBER,
    CATEGORY_SSN,
    FakePIIDetector,
    LocalRegexPIIDetector,
    PIIEntity,
    _luhn_is_valid,
    _routing_checksum_is_valid,
    get_pii_detector,
)


def _categories(text, detector=None):
    detector = detector or LocalRegexPIIDetector()
    return [e.category for e in detector.detect(text)]


def _matched(text, detector=None):
    detector = detector or LocalRegexPIIDetector()
    return [(e.category, text[e.start : e.end]) for e in detector.detect(text)]


# --- SSN -----------------------------------------------------------------------


def test_formatted_ssn_is_detected():
    assert _matched("My SSN on file is 123-45-6789 for verification.") == [(CATEGORY_SSN, "123-45-6789")]


def test_unformatted_ssn_requires_context():
    assert _matched("SSN: 123456789 is on record.") == [(CATEGORY_SSN, "123456789")]


def test_bare_nine_digits_without_context_are_not_masked_as_ssn():
    # No "SSN"/"social security" context nearby -- an ambiguous 9-digit
    # run must not be masked (avoids over-masking e.g. a case/order number).
    assert _categories("Reference number 123456789 was issued.") == []


@pytest.mark.parametrize(
    "text",
    [
        "SSN: 000-45-6789 is on file.",  # area 000
        "SSN: 666-45-6789 is on file.",  # area 666
        "SSN: 900-45-6789 is on file.",  # area 900-999
        "SSN: 999-45-6789 is on file.",  # area 900-999 (upper bound)
        "SSN: 123-00-6789 is on file.",  # group 00
        "SSN: 123-45-0000 is on file.",  # serial 0000
    ],
)
def test_ssn_with_invalid_group_is_never_masked(text):
    # Structurally impossible SSNs (per SSA rules) must not be treated as
    # real ones, even though they match the XXX-XX-XXXX shape.
    assert _categories(text) == []


def test_ssn_with_invalid_group_is_not_masked_even_with_explicit_context():
    assert _categories("Social Security Number: 666456789 is on record.") == []


# --- Card numbers (Luhn) ---------------------------------------------------------


def test_valid_luhn_card_formatted_is_detected():
    assert _matched("Card 4111 1111 1111 1111 on file.") == [(CATEGORY_CARD, "4111 1111 1111 1111")]


def test_valid_luhn_card_unformatted_is_detected():
    assert _matched("Card 4111111111111111 on file.") == [(CATEGORY_CARD, "4111111111111111")]


def test_valid_luhn_card_dash_formatted_is_detected():
    assert _matched("Card 4111-1111-1111-1111 on file.") == [(CATEGORY_CARD, "4111-1111-1111-1111")]


def test_invalid_luhn_card_is_not_masked():
    # Same shape as a real card, last digit changed to break the checksum.
    assert _categories("Card 4111 1111 1111 1112 on file.") == []


@pytest.mark.parametrize(
    ("digits", "expected"),
    [
        ("4111111111111111", True),  # well-known Luhn-valid test number
        ("4111111111111112", False),
        ("79927398713", True),  # classic Luhn test vector
        ("79927398710", False),
    ],
)
def test_luhn_validation_function(digits, expected):
    assert _luhn_is_valid(digits) is expected


# --- Email -----------------------------------------------------------------------


def test_email_is_detected():
    assert _matched("Contact john.doe+billing@example.co.uk for help.") == [
        (CATEGORY_EMAIL, "john.doe+billing@example.co.uk")
    ]


def test_email_masking_can_be_disabled():
    detector = LocalRegexPIIDetector(mask_emails=False)
    assert _categories("Contact john@example.com now.", detector) == []


# --- Phone -------------------------------------------------------------------------


def test_formatted_phone_variants_are_detected():
    for text in ["(555) 123-4567", "555-123-4567", "555.123.4567", "+1 555-123-4567"]:
        assert _categories(f"Call {text} today.") == [CATEGORY_PHONE], text


def test_unformatted_phone_is_detected():
    assert _matched("Call 5551234567 today.") == [(CATEGORY_PHONE, "5551234567")]


def test_phone_masking_can_be_disabled():
    detector = LocalRegexPIIDetector(mask_phones=False)
    assert _categories("Call 555-123-4567 today.", detector) == []


# --- IPv4 --------------------------------------------------------------------------


def test_ipv4_is_detected():
    assert _matched("Server at 192.168.1.1 handled the request.") == [(CATEGORY_IPV4, "192.168.1.1")]


def test_ipv4_octet_boundaries_are_respected():
    # 999 is not a valid octet -- must not be matched as (part of) an IPv4.
    assert _categories("Not an IP: 999.999.999.999") == []


# --- Date of birth (context-gated) --------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Date of Birth: 01/15/1985 is on file.",
        "DOB: 1985-01-15 recorded.",
        "Born on January 15, 1985 in the city.",
    ],
)
def test_labeled_dob_is_detected(text):
    assert _categories(text) == [CATEGORY_DOB]


def test_ordinary_date_without_dob_context_is_not_masked():
    assert _categories("Statement Date: 01/15/2024 for this period.") == []
    assert _categories("Effective Date: 03/01/2024 going forward.") == []


@pytest.mark.parametrize(
    "text",
    [
        "Date of Birth: 13/45/2020 is on file.",  # month 13, day 45 -- impossible either way
        "Date of Birth: 02/30/2020 is on file.",  # Feb 30 does not exist
        "Date of Birth: 00/15/1985 is on file.",  # month 00 is not valid
        "Date of Birth: 04/31/1990 is on file.",  # April has 30 days
    ],
)
def test_impossible_dob_date_is_never_masked(text):
    # Date-shaped but calendar-impossible under any month/day reading --
    # must not be masked as a DOB even though the DOB context is present.
    assert _categories(text) == []


def test_valid_dob_with_dot_abbreviated_month_is_still_detected():
    assert _categories("Born on Jan. 15, 1985 in the city.") == [CATEGORY_DOB]


# --- Bank account (context-gated) ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Account Number: 000123456789 on file.",
        "Acct No 987654321 is active.",
        "A/C #: 1234567890123 for deposits.",
    ],
)
def test_context_gated_account_number_is_detected(text):
    assert _categories(text) == [CATEGORY_BANK_ACCOUNT]


def test_bare_digits_without_account_context_are_not_masked():
    assert _categories("The invoice total is 000123456789 units.") == []


def test_financial_identifier_masking_can_be_disabled_for_accounts():
    detector = LocalRegexPIIDetector(mask_financial_identifiers=False)
    assert _categories("Account Number: 000123456789 on file.", detector) == []


# --- Routing number (context + checksum) ---------------------------------------------


def test_valid_routing_number_with_context_is_detected():
    # 021000021 is a real, published Chase NY ABA routing number (valid checksum).
    assert _matched("Routing Number: 021000021 for wire transfers.") == [
        (CATEGORY_ROUTING_NUMBER, "021000021")
    ]


def test_invalid_routing_checksum_is_not_masked_even_with_context():
    assert _categories("Routing Number: 123456789 for wire transfers.") == []


def test_valid_routing_digits_without_context_are_not_masked():
    assert _categories("Some code: 021000021 appeared here.") == []


@pytest.mark.parametrize(
    ("digits", "expected"),
    [
        ("021000021", True),  # Chase NY
        ("011401533", True),  # Bank of America MA (published, valid)
        ("123456789", False),
        ("000000000", True),  # degenerate but checksum-valid (0 mod 10 == 0)
        ("12345678", False),  # wrong length
    ],
)
def test_routing_checksum_function(digits, expected):
    assert _routing_checksum_is_valid(digits) is expected


# --- False-positive avoidance (explicitly required) -----------------------------------


def test_dollar_amounts_are_never_masked():
    assert _categories("Total due: $1,000.00 by the due date.") == []


def test_interest_rates_and_percentages_are_never_masked():
    assert _categories("APR is 12.5% and the penalty rate is 29.99%.") == []


def test_policy_numbers_without_sensitive_context_are_never_masked():
    assert _categories("Policy number POL-2024-88291 is referenced below.") == []


def test_page_numbers_are_never_masked():
    assert _categories("See page 3 of 12 for details.") == []


def test_zip_codes_are_never_masked():
    assert _categories("Mail to Springfield, IL 62704 or ZIP+4 62704-1234.") == []


def test_general_financial_statistics_are_never_masked():
    assert _categories("Average balance grew 4.2% to $12,345 across 1,024 accounts.") == []


# --- Overlap resolution / determinism / duplicates -------------------------------------


def test_duplicate_values_are_each_detected_at_their_own_position():
    text = "SSN 123-45-6789 repeated: 123-45-6789."
    matches = _matched(text)
    assert matches == [(CATEGORY_SSN, "123-45-6789"), (CATEGORY_SSN, "123-45-6789")]


def test_detection_is_deterministic_across_repeated_runs():
    text = "Email a@example.com, phone 555-123-4567, SSN 123-45-6789, card 4111 1111 1111 1111."
    detector = LocalRegexPIIDetector()
    first = [(e.category, e.start, e.end) for e in detector.detect(text)]
    second = [(e.category, e.start, e.end) for e in detector.detect(text)]
    assert first == second
    assert len(first) == 4


def test_entities_are_returned_sorted_and_non_overlapping():
    text = "Email a@example.com, phone 555-123-4567, SSN 123-45-6789."
    entities = LocalRegexPIIDetector().detect(text)
    starts = [e.start for e in entities]
    assert starts == sorted(starts)
    for a, b in zip(entities, entities[1:]):
        assert a.end <= b.start


# --- Unicode surrounding text --------------------------------------------------------


def test_unicode_surrounding_text_does_not_break_detection():
    text = "Kontoinhaber Müller — Account Number: 123456789012 ist wichtig. 🏦 E-Mail test@example.com"
    matches = _matched(text)
    assert (CATEGORY_BANK_ACCOUNT, "123456789012") in matches
    assert (CATEGORY_EMAIL, "test@example.com") in matches


def test_unicode_multiline_text_keeps_offsets_consistent():
    text = (
        "Kontoauszug — Übersicht\n"
        "Zeile mit Emoji 🏦🏦\n"
        "SSN: 123-45-6789 noted.\n"
        "Weitere Zeile mit Ümlaut äöü\n"
        "Email: test@example.com für Rückfragen.\n"
    )
    matches = _matched(text)
    assert (CATEGORY_SSN, "123-45-6789") in matches
    assert (CATEGORY_EMAIL, "test@example.com") in matches


def test_unicode_astral_characters_keep_offsets_consistent():
    # Emoji before the entity shifts code-point offsets; matched substring
    # must still be exactly the entity, proving start/end stay correct.
    text = "🏦🏦🏦 SSN: 123-45-6789 noted."
    matches = _matched(text)
    assert matches == [(CATEGORY_SSN, "123-45-6789")]


# --- apply_masking / overlap-resolution priority ----------------------------------------


def test_overlap_resolution_prefers_higher_priority_category():
    from app.services.privacy.detectors import _resolve_overlaps

    # Two overlapping candidates over the same span -- SSN must win over
    # a lower-priority, looser category by the documented priority order.
    entities = [
        PIIEntity(category=CATEGORY_PHONE, start=0, end=10, placeholder="[PHONE_REDACTED]"),
        PIIEntity(category=CATEGORY_SSN, start=0, end=10, placeholder="[SSN_REDACTED]"),
    ]
    resolved = _resolve_overlaps(entities)
    assert len(resolved) == 1
    assert resolved[0].category == CATEGORY_SSN


def test_card_number_overlapping_account_context_resolves_to_card():
    # A Luhn-valid card number that also happens to sit right after
    # "Account Number:" produces two overlapping candidates (CARD and
    # BANK_ACCOUNT) over the identical span -- CARD is the higher-priority,
    # more specific (checksum-validated) category and must win.
    text = "Account Number: 4111111111111111 was charged."
    assert _categories(text) == [CATEGORY_CARD]


def test_ssn_context_wins_over_routing_context_for_the_identical_span():
    # A single 9-digit value ("011401533" -- a real, checksum-valid ABA
    # routing number) sitting in a window reachable by both SSN and
    # ROUTING_NUMBER context keywords produces two overlapping candidates
    # over the exact same span. SSN is the higher-priority category and
    # must be the one that wins -- not a double-masked/duplicated result.
    text = "SSN Routing Number: 011401533 for wire transfers."
    matches = _matched(text)
    assert matches == [(CATEGORY_SSN, "011401533")]


def test_routing_number_overlapping_bare_nine_digits_is_not_double_masked():
    # The same 9 digits could, in principle, also satisfy the SSN
    # unformatted pattern if SSN context were present -- with only
    # routing context present, exactly one entity (ROUTING_NUMBER) must
    # be produced, not two overlapping ones.
    text = "Routing Number: 021000021 for wire transfers."
    matches = _matched(text)
    assert matches == [(CATEGORY_ROUTING_NUMBER, "021000021")]


def test_adjacent_values_with_no_separator_never_produce_overlapping_entities():
    # Two email-shaped runs jammed together with no separator between them
    # -- whatever the regex greedily consumes, entities must never overlap
    # or double-count the same span.
    text = "a@example.combob@example.org"
    entities = LocalRegexPIIDetector().detect(text)
    assert len(entities) == 1
    assert entities[0].category == CATEGORY_EMAIL
    for a, b in zip(entities, entities[1:]):
        assert a.end <= b.start


# --- Idempotency: masked/placeholder text is never re-detected -----------------------


@pytest.mark.parametrize(
    "placeholder_text",
    [
        "[SSN_REDACTED]",
        "[CARD_REDACTED]",
        "[EMAIL_REDACTED]",
        "[PHONE_REDACTED]",
        "[IP_REDACTED]",
        "[DOB_REDACTED]",
        "[ACCOUNT_REDACTED]",
        "[ROUTING_REDACTED]",
        "Contact [EMAIL_REDACTED] regarding SSN [SSN_REDACTED] and card [CARD_REDACTED].",
    ],
)
def test_placeholder_text_is_never_itself_detected_as_pii(placeholder_text):
    assert LocalRegexPIIDetector().detect(placeholder_text) == []


def test_masking_already_masked_text_is_a_no_op():
    from app.services.privacy.masking import apply_masking

    text = "Card 4111 1111 1111 1111 on file, SSN 123-45-6789 too."
    detector = LocalRegexPIIDetector()
    once = apply_masking(text, detector.detect(text))
    twice = apply_masking(once, detector.detect(once))
    assert once == twice


# --- FakePIIDetector -------------------------------------------------------------------


def test_fake_pii_detector_returns_canned_entities():
    entity = PIIEntity(category=CATEGORY_SSN, start=0, end=3, placeholder="[SSN_REDACTED]")
    detector = FakePIIDetector(entities=[entity])
    assert detector.detect("anything") == [entity]


def test_fake_pii_detector_can_raise():
    detector = FakePIIDetector(raise_exception=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        detector.detect("anything")


# --- get_pii_detector() dependency factory ------------------------------------------------


def test_get_pii_detector_returns_a_real_local_detector():
    get_pii_detector.cache_clear()
    try:
        detector = get_pii_detector()
        assert isinstance(detector, LocalRegexPIIDetector)
    finally:
        get_pii_detector.cache_clear()
