import pytest
from agent_guard_rails import (
    Violation, GuardResult, GuardRailsResult, GuardRails, check,
    MaxLength, MinLength, RequiredPhrase, ForbiddenPhrase,
    MatchesRegex, NoRegex, CustomGuard, AllOfGuard, AnyOfGuard, NotGuard,
)


# ---------------------------------------------------------------------------
# MaxLength
# ---------------------------------------------------------------------------

def test_max_length_ok():
    assert MaxLength(100).check("hello").ok

def test_max_length_fail():
    assert not MaxLength(3).check("hello").ok

def test_max_length_exact():
    assert MaxLength(5).check("hello").ok

def test_max_length_violation_message():
    r = MaxLength(3).check("hello")
    assert "5" in r.violations[0].message

def test_max_length_name():
    assert "100" in MaxLength(100).name


# ---------------------------------------------------------------------------
# MinLength
# ---------------------------------------------------------------------------

def test_min_length_ok():
    assert MinLength(3).check("hello").ok

def test_min_length_fail():
    assert not MinLength(10).check("hello").ok

def test_min_length_exact():
    assert MinLength(5).check("hello").ok

def test_min_length_violation_message():
    r = MinLength(10).check("hello")
    assert "5" in r.violations[0].message


# ---------------------------------------------------------------------------
# RequiredPhrase
# ---------------------------------------------------------------------------

def test_required_phrase_ok():
    assert RequiredPhrase("DONE").check("Task is DONE").ok

def test_required_phrase_fail():
    assert not RequiredPhrase("DONE").check("not finished").ok

def test_required_phrase_case_insensitive():
    assert RequiredPhrase("done").check("TASK DONE").ok

def test_required_phrase_case_sensitive():
    assert not RequiredPhrase("DONE", case_sensitive=True).check("task done").ok


# ---------------------------------------------------------------------------
# ForbiddenPhrase
# ---------------------------------------------------------------------------

def test_forbidden_phrase_ok():
    assert ForbiddenPhrase("sorry").check("Here is the answer.").ok

def test_forbidden_phrase_fail():
    assert not ForbiddenPhrase("sorry").check("I am sorry but I cannot.").ok

def test_forbidden_phrase_case_insensitive():
    assert not ForbiddenPhrase("sorry").check("SORRY for that").ok

def test_forbidden_phrase_case_sensitive():
    assert ForbiddenPhrase("SORRY", case_sensitive=True).check("sorry about that").ok


# ---------------------------------------------------------------------------
# MatchesRegex
# ---------------------------------------------------------------------------

def test_matches_regex_ok():
    assert MatchesRegex(r"\d+").check("answer is 42").ok

def test_matches_regex_fail():
    assert not MatchesRegex(r"\d+").check("no digits here").ok

def test_matches_regex_violation_message():
    r = MatchesRegex(r"\d+").check("no digits")
    assert r.violations

def test_matches_regex_flags():
    assert MatchesRegex(r"hello", flags=0).check("hello world").ok


# ---------------------------------------------------------------------------
# NoRegex
# ---------------------------------------------------------------------------

def test_no_regex_ok():
    assert NoRegex(r"ABORT").check("continue working").ok

def test_no_regex_fail():
    assert not NoRegex(r"ABORT").check("ABORT task").ok

def test_no_regex_violation_message():
    r = NoRegex(r"ABORT").check("ABORT task")
    assert "ABORT" in r.violations[0].message


# ---------------------------------------------------------------------------
# CustomGuard
# ---------------------------------------------------------------------------

def test_custom_guard_true():
    g = CustomGuard(lambda t: True)
    assert g.check("anything").ok

def test_custom_guard_false():
    g = CustomGuard(lambda t: False)
    assert not g.check("anything").ok

def test_custom_guard_none():
    g = CustomGuard(lambda t: None)
    assert g.check("anything").ok

def test_custom_guard_string():
    g = CustomGuard(lambda t: "bad output")
    r = g.check("x")
    assert not r.ok
    assert "bad output" in r.violations[0].message

def test_custom_guard_guard_result():
    from agent_guard_rails import GuardResult, Violation
    g = CustomGuard(lambda t: GuardResult(ok=False, violations=[Violation("c", "msg")]))
    r = g.check("x")
    assert not r.ok

def test_custom_guard_exception():
    g = CustomGuard(lambda t: 1 / 0)
    r = g.check("x")
    assert not r.ok
    assert "division by zero" in r.violations[0].message

def test_custom_guard_name():
    g = CustomGuard(lambda t: True, name="my_check")
    assert g.name == "my_check"


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------

def test_all_of_pass():
    g = AllOfGuard([MaxLength(100), MinLength(1)])
    assert g.check("hello").ok

def test_all_of_fail_one():
    g = AllOfGuard([MaxLength(3), MinLength(1)])
    assert not g.check("hello").ok

def test_all_of_multiple_violations():
    g = AllOfGuard([MaxLength(3), RequiredPhrase("DONE")])
    r = g.check("hello world")
    assert len(r.violations) == 2

def test_any_of_pass_one():
    g = AnyOfGuard([MaxLength(3), MaxLength(100)])
    assert g.check("hello").ok

def test_any_of_all_fail():
    g = AnyOfGuard([MaxLength(1), MaxLength(2)])
    assert not g.check("hello").ok

def test_not_guard_inverts():
    g = NotGuard(MaxLength(1))   # fail when <= 1 char
    assert g.check("hello").ok   # hello > 1 so MaxLength(1) fails → Not passes

def test_not_guard_inverts_passes():
    g = NotGuard(MaxLength(100))  # MaxLength(100) passes → Not fails
    assert not g.check("hello").ok

def test_operator_and():
    g = MaxLength(100) & MinLength(1)
    assert isinstance(g, AllOfGuard)
    assert g.check("hello").ok

def test_operator_or():
    g = MaxLength(3) | MaxLength(100)
    assert isinstance(g, AnyOfGuard)
    assert g.check("hello").ok

def test_operator_invert():
    g = ~MaxLength(1)
    assert isinstance(g, NotGuard)


# ---------------------------------------------------------------------------
# GuardRails
# ---------------------------------------------------------------------------

def test_guard_rails_all_pass():
    rails = GuardRails([MaxLength(100), MinLength(1)])
    r = rails.check("hello")
    assert r.ok

def test_guard_rails_violation():
    rails = GuardRails([MaxLength(3)])
    r = rails.check("hello")
    assert not r.ok
    assert len(r.violations) == 1

def test_guard_rails_multiple_violations():
    rails = GuardRails([MaxLength(3), RequiredPhrase("DONE")])
    r = rails.check("hello world")
    assert r.violation_count == 2

def test_guard_rails_failed_rules():
    rails = GuardRails([MaxLength(3)])
    r = rails.check("hello")
    assert len(r.failed_rules) == 1

def test_guard_rails_bool():
    rails = GuardRails([MaxLength(100)])
    assert bool(rails.check("hello"))

def test_guard_rails_repr():
    rails = GuardRails([MaxLength(3)])
    r = rails.check("hi")
    assert "True" in repr(r) or "False" in repr(r)

def test_guard_rails_add():
    rails = GuardRails([MaxLength(100)])
    rails.add(MinLength(1))
    assert len(rails) == 2

def test_guard_rails_len():
    rails = GuardRails([MaxLength(100), MinLength(1)])
    assert len(rails) == 2

def test_module_check():
    r = check("hello world", MaxLength(100), MinLength(1))
    assert r.ok

def test_violation_str():
    v = Violation("rule", "message")
    assert "rule" in str(v)
    assert "message" in str(v)
