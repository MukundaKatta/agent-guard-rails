"""Tests for agent-guard-rails."""

import pytest

from agent_guard_rails import (
    BannedPhrase,
    GuardRails,
    GuardRailViolationError,
    MaxLength,
    MinLength,
    RegexMustMatch,
    RegexMustNotMatch,
    RequiredPhrase,
)

# ---------------------------------------------------------------------------
# MaxLength
# ---------------------------------------------------------------------------


def test_max_length_pass():
    rail = MaxLength(10)
    assert rail.check("hello") == []


def test_max_length_exact_boundary_pass():
    rail = MaxLength(5)
    assert rail.check("hello") == []


def test_max_length_fail():
    rail = MaxLength(5)
    violations = rail.check("toolong!")
    assert len(violations) == 1
    assert "8 chars, max 5" in violations[0]


def test_max_length_violation_message():
    rail = MaxLength(3)
    violations = rail.check("abcd")
    assert violations == ["Response too long: 4 chars, max 3"]


# ---------------------------------------------------------------------------
# MinLength
# ---------------------------------------------------------------------------


def test_min_length_pass():
    rail = MinLength(3)
    assert rail.check("hello") == []


def test_min_length_fail():
    rail = MinLength(10)
    violations = rail.check("hi")
    assert len(violations) == 1
    assert "2 chars, min 10" in violations[0]


def test_min_length_strips_whitespace():
    rail = MinLength(5)
    # "hi" stripped is 2 chars — should fail even with surrounding spaces
    violations = rail.check("  hi  ")
    assert len(violations) == 1
    assert "2 chars, min 5" in violations[0]


def test_min_length_violation_message():
    rail = MinLength(5)
    violations = rail.check("ab")
    assert violations == ["Response too short: 2 chars, min 5"]


# ---------------------------------------------------------------------------
# BannedPhrase
# ---------------------------------------------------------------------------


def test_banned_phrase_single_found():
    rail = BannedPhrase("badword")
    violations = rail.check("This contains badword here.")
    assert len(violations) == 1
    assert "badword" in violations[0]


def test_banned_phrase_not_found():
    rail = BannedPhrase("badword")
    assert rail.check("This is fine.") == []


def test_banned_phrase_multiple_only_one_found():
    rail = BannedPhrase("alpha", "beta", "gamma")
    violations = rail.check("Only beta is here.")
    assert len(violations) == 1
    assert "beta" in violations[0]


def test_banned_phrase_case_insensitive_default():
    rail = BannedPhrase("BadWord")
    violations = rail.check("This has BADWORD in it.")
    assert len(violations) == 1


def test_banned_phrase_case_sensitive_misses_wrong_case():
    rail = BannedPhrase("BadWord", case_sensitive=True)
    assert rail.check("This has badword in it.") == []


def test_banned_phrase_violation_message_format():
    rail = BannedPhrase("oops")
    violations = rail.check("oops I did it again")
    assert violations == ["Banned phrase found: 'oops'"]


# ---------------------------------------------------------------------------
# RequiredPhrase
# ---------------------------------------------------------------------------


def test_required_phrase_all_present():
    rail = RequiredPhrase("hello", "world")
    assert rail.check("hello there world") == []


def test_required_phrase_one_missing():
    rail = RequiredPhrase("hello", "world")
    violations = rail.check("hello there")
    assert len(violations) == 1
    assert "world" in violations[0]


def test_required_phrase_case_insensitive_default():
    rail = RequiredPhrase("Hello")
    assert rail.check("say hello there") == []


def test_required_phrase_case_sensitive_misses():
    rail = RequiredPhrase("Hello", case_sensitive=True)
    violations = rail.check("say hello there")
    assert len(violations) == 1


def test_required_phrase_violation_message_format():
    rail = RequiredPhrase("disclaimer")
    violations = rail.check("No disclaimer here")
    # "disclaimer" is present — should pass
    assert violations == []


def test_required_phrase_missing_message_format():
    rail = RequiredPhrase("LEGAL_NOTICE")
    violations = rail.check("Response without it.")
    assert violations == ["Required phrase missing: 'LEGAL_NOTICE'"]


# ---------------------------------------------------------------------------
# RegexMustMatch
# ---------------------------------------------------------------------------


def test_regex_must_match_pass():
    rail = RegexMustMatch(r"\d{3}")
    assert rail.check("Code 123 here") == []


def test_regex_must_match_fail_default_message():
    rail = RegexMustMatch(r"\d{3}")
    violations = rail.check("No numbers here")
    assert len(violations) == 1
    assert r"\d{3}" in violations[0]


def test_regex_must_match_custom_message():
    rail = RegexMustMatch(r"\d+", message="Needs a number")
    violations = rail.check("No digits")
    assert violations == ["Needs a number"]


# ---------------------------------------------------------------------------
# RegexMustNotMatch
# ---------------------------------------------------------------------------


def test_regex_must_not_match_pass():
    rail = RegexMustNotMatch(r"<script>")
    assert rail.check("Safe text here") == []


def test_regex_must_not_match_fail_default_message():
    rail = RegexMustNotMatch(r"<script>")
    violations = rail.check("Inject <script>alert(1)</script>")
    assert len(violations) == 1
    assert "<script>" in violations[0]


def test_regex_must_not_match_custom_message():
    rail = RegexMustNotMatch(r"NSFW", message="Content policy violated")
    violations = rail.check("NSFW content detected")
    assert violations == ["Content policy violated"]


# ---------------------------------------------------------------------------
# GuardRails — empty / basic
# ---------------------------------------------------------------------------


def test_guard_rails_empty_always_passes():
    gr = GuardRails()
    result = gr.check("anything at all")
    assert result.passed is True
    assert result.violations == []


def test_guard_rails_multiple_rails_all_violations_collected():
    gr = GuardRails([MaxLength(3), MinLength(10)])
    result = gr.check("hi")
    # MaxLength(3) passes (2 <= 3), MinLength(10) fails (2 < 10)
    assert result.passed is False
    assert len(result.violations) == 1

    result2 = gr.check("toolongtext")
    # MaxLength(3) fails (11 > 3), MinLength(10) passes (11 >= 10)
    assert result2.passed is False
    assert len(result2.violations) == 1


def test_guard_rails_collects_all_violations_at_once():
    gr = GuardRails([BannedPhrase("bad"), BannedPhrase("evil")])
    result = gr.check("This is bad and evil.")
    assert result.passed is False
    assert len(result.violations) == 2


def test_guard_rails_check_passed_true():
    gr = GuardRails([MaxLength(100), MinLength(1)])
    result = gr.check("Short but valid.")
    assert result.passed is True


def test_guard_rails_check_passed_false():
    gr = GuardRails([MaxLength(2)])
    result = gr.check("too long")
    assert result.passed is False


# ---------------------------------------------------------------------------
# enforce()
# ---------------------------------------------------------------------------


def test_enforce_returns_text_on_pass():
    gr = GuardRails([MaxLength(100)])
    text = "Hello world"
    assert gr.enforce(text) is text


def test_enforce_raises_on_fail():
    gr = GuardRails([MaxLength(3)])
    with pytest.raises(GuardRailViolationError):
        gr.enforce("too long text")


def test_guard_rail_violation_error_violations_attribute():
    gr = GuardRails([MaxLength(3), BannedPhrase("x")])
    try:
        gr.enforce("x toolong")
    except GuardRailViolationError as exc:
        assert isinstance(exc.violations, list)
        assert len(exc.violations) == 2


# ---------------------------------------------------------------------------
# add() / remove() / rails()
# ---------------------------------------------------------------------------


def test_add_returns_self_for_chaining():
    gr = GuardRails()
    result = gr.add(MaxLength(100)).add(MinLength(1))
    assert result is gr
    assert len(gr.rails()) == 2


def test_remove_removes_correct_index():
    r1, r2, r3 = MaxLength(10), MinLength(1), BannedPhrase("x")
    gr = GuardRails([r1, r2, r3])
    gr.remove(1)
    remaining = gr.rails()
    assert remaining == [r1, r3]


def test_remove_raises_index_error_on_bad_index():
    gr = GuardRails([MaxLength(10)])
    with pytest.raises(IndexError):
        gr.remove(5)


def test_rails_returns_copy():
    r = MaxLength(50)
    gr = GuardRails([r])
    copy = gr.rails()
    copy.append(MinLength(1))  # mutate the copy
    # original should be unaffected
    assert len(gr.rails()) == 1


# ---------------------------------------------------------------------------
# guarded() decorator
# ---------------------------------------------------------------------------


def test_guarded_passes_through_clean_output():
    gr = GuardRails([MaxLength(100)])

    @gr.guarded()
    def generate():
        return "Short clean response."

    assert generate() == "Short clean response."


def test_guarded_raises_on_violation():
    gr = GuardRails([MaxLength(5)])

    @gr.guarded()
    def generate():
        return "This output is way too long"

    with pytest.raises(GuardRailViolationError):
        generate()
