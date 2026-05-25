"""
agent_guard_rails — composable output guardrails for LLM responses.

Define rules (length, phrases, regex, custom), apply them to text,
and get a structured violation report. Zero dependencies (stdlib: re, dataclasses).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    """A single rule violation."""

    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


@dataclass
class GuardResult:
    """Result of checking a single guardrail."""

    ok: bool
    violations: list[Violation] = field(default_factory=list)
    rule: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class GuardRailsResult:
    """Aggregated result from running all guardrails."""

    ok: bool
    violations: list[Violation]
    text: str

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def failed_rules(self) -> list[str]:
        return [v.rule for v in self.violations]

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return f"GuardRailsResult(ok={self.ok}, violations={self.violation_count})"


# ---------------------------------------------------------------------------
# Base guardrail
# ---------------------------------------------------------------------------

class Guardrail(ABC):
    """Base class for a single guardrail rule."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable rule name."""

    @abstractmethod
    def check(self, text: str) -> GuardResult:
        """Check *text* against this rule."""

    def __call__(self, text: str) -> GuardResult:
        return self.check(text)

    def __and__(self, other: Guardrail) -> "AllOfGuard":
        return AllOfGuard([self, other])

    def __or__(self, other: Guardrail) -> "AnyOfGuard":
        return AnyOfGuard([self, other])

    def __invert__(self) -> "NotGuard":
        return NotGuard(self)


# ---------------------------------------------------------------------------
# Built-in guardrails
# ---------------------------------------------------------------------------

class MaxLength(Guardrail):
    """Text must not exceed *max_chars* characters."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    @property
    def name(self) -> str:
        return f"MaxLength({self.max_chars})"

    def check(self, text: str) -> GuardResult:
        if len(text) <= self.max_chars:
            return GuardResult(ok=True, rule=self.name)
        return GuardResult(
            ok=False,
            violations=[Violation(self.name, f"text is {len(text)} chars, max {self.max_chars}")],
            rule=self.name,
        )


class MinLength(Guardrail):
    """Text must be at least *min_chars* characters."""

    def __init__(self, min_chars: int) -> None:
        self.min_chars = min_chars

    @property
    def name(self) -> str:
        return f"MinLength({self.min_chars})"

    def check(self, text: str) -> GuardResult:
        if len(text) >= self.min_chars:
            return GuardResult(ok=True, rule=self.name)
        return GuardResult(
            ok=False,
            violations=[Violation(self.name, f"text is {len(text)} chars, min {self.min_chars}")],
            rule=self.name,
        )


class RequiredPhrase(Guardrail):
    """Text must contain *phrase*."""

    def __init__(self, phrase: str, *, case_sensitive: bool = False) -> None:
        self.phrase = phrase
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return f"RequiredPhrase({self.phrase!r})"

    def check(self, text: str) -> GuardResult:
        haystack = text if self.case_sensitive else text.lower()
        needle = self.phrase if self.case_sensitive else self.phrase.lower()
        if needle in haystack:
            return GuardResult(ok=True, rule=self.name)
        return GuardResult(
            ok=False,
            violations=[Violation(self.name, f"required phrase {self.phrase!r} not found")],
            rule=self.name,
        )


class ForbiddenPhrase(Guardrail):
    """Text must NOT contain *phrase*."""

    def __init__(self, phrase: str, *, case_sensitive: bool = False) -> None:
        self.phrase = phrase
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return f"ForbiddenPhrase({self.phrase!r})"

    def check(self, text: str) -> GuardResult:
        haystack = text if self.case_sensitive else text.lower()
        needle = self.phrase if self.case_sensitive else self.phrase.lower()
        if needle not in haystack:
            return GuardResult(ok=True, rule=self.name)
        return GuardResult(
            ok=False,
            violations=[Violation(self.name, f"forbidden phrase {self.phrase!r} found")],
            rule=self.name,
        )


class MatchesRegex(Guardrail):
    """Text must match *pattern* (search, not full match)."""

    def __init__(self, pattern: str, *, flags: int = re.IGNORECASE) -> None:
        self.pattern = pattern
        self.flags = flags
        self._re = re.compile(pattern, flags)

    @property
    def name(self) -> str:
        return f"MatchesRegex({self.pattern!r})"

    def check(self, text: str) -> GuardResult:
        if self._re.search(text):
            return GuardResult(ok=True, rule=self.name)
        return GuardResult(
            ok=False,
            violations=[Violation(self.name, f"pattern {self.pattern!r} not found in text")],
            rule=self.name,
        )


class NoRegex(Guardrail):
    """Text must NOT match *pattern*."""

    def __init__(self, pattern: str, *, flags: int = re.IGNORECASE) -> None:
        self.pattern = pattern
        self.flags = flags
        self._re = re.compile(pattern, flags)

    @property
    def name(self) -> str:
        return f"NoRegex({self.pattern!r})"

    def check(self, text: str) -> GuardResult:
        m = self._re.search(text)
        if not m:
            return GuardResult(ok=True, rule=self.name)
        return GuardResult(
            ok=False,
            violations=[Violation(self.name, f"forbidden pattern {self.pattern!r} matched: {m.group()!r}")],
            rule=self.name,
        )


class CustomGuard(Guardrail):
    """Wrap any callable as a guardrail.

    The function receives *text* and should return:
    - ``True`` / ``None`` → OK
    - ``False`` → FAIL
    - A string → FAIL with that message
    - A :class:`GuardResult` directly
    """

    def __init__(self, fn: Callable[[str], Any], *, name: str = "custom") -> None:
        self._fn = fn
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def check(self, text: str) -> GuardResult:
        try:
            result = self._fn(text)
        except Exception as exc:
            return GuardResult(
                ok=False,
                violations=[Violation(self.name, str(exc))],
                rule=self.name,
            )
        if isinstance(result, GuardResult):
            return result
        if result is True or result is None:
            return GuardResult(ok=True, rule=self.name)
        if result is False:
            return GuardResult(ok=False, violations=[Violation(self.name, "check failed")], rule=self.name)
        if isinstance(result, str):
            return GuardResult(ok=False, violations=[Violation(self.name, result)], rule=self.name)
        return GuardResult(ok=True, rule=self.name)


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------

class AllOfGuard(Guardrail):
    """All guards must pass."""

    def __init__(self, guards: list[Guardrail]) -> None:
        self._guards = guards

    @property
    def name(self) -> str:
        return f"AllOf({', '.join(g.name for g in self._guards)})"

    def check(self, text: str) -> GuardResult:
        violations: list[Violation] = []
        for g in self._guards:
            r = g.check(text)
            violations.extend(r.violations)
        return GuardResult(ok=len(violations) == 0, violations=violations, rule=self.name)


class AnyOfGuard(Guardrail):
    """At least one guard must pass."""

    def __init__(self, guards: list[Guardrail]) -> None:
        self._guards = guards

    @property
    def name(self) -> str:
        return f"AnyOf({', '.join(g.name for g in self._guards)})"

    def check(self, text: str) -> GuardResult:
        for g in self._guards:
            r = g.check(text)
            if r.ok:
                return GuardResult(ok=True, rule=self.name)
        # All failed
        violations = [Violation(self.name, "none of the guards passed")]
        return GuardResult(ok=False, violations=violations, rule=self.name)


class NotGuard(Guardrail):
    """Invert a guard — passes when the wrapped guard fails."""

    def __init__(self, guard: Guardrail) -> None:
        self._guard = guard

    @property
    def name(self) -> str:
        return f"Not({self._guard.name})"

    def check(self, text: str) -> GuardResult:
        r = self._guard.check(text)
        if not r.ok:
            return GuardResult(ok=True, rule=self.name)
        return GuardResult(
            ok=False,
            violations=[Violation(self.name, f"expected {self._guard.name!r} to fail but it passed")],
            rule=self.name,
        )


# ---------------------------------------------------------------------------
# GuardRails runner
# ---------------------------------------------------------------------------

class GuardRails:
    """
    Run a set of guardrails against LLM output.

    Usage::

        rails = GuardRails([
            MaxLength(2000),
            MinLength(50),
            RequiredPhrase("DONE"),
            ForbiddenPhrase("I don't know"),
        ])

        result = rails.check(response_text)
        if not result.ok:
            for v in result.violations:
                print(v)
    """

    def __init__(self, guards: list[Guardrail]) -> None:
        self._guards = list(guards)

    def add(self, guard: Guardrail) -> None:
        """Add a guardrail."""
        self._guards.append(guard)

    def check(self, text: str) -> GuardRailsResult:
        """
        Run all guardrails against *text*.

        Args:
            text: LLM output text to check.

        Returns:
            :class:`GuardRailsResult`
        """
        all_violations: list[Violation] = []
        for g in self._guards:
            r = g.check(text)
            all_violations.extend(r.violations)
        return GuardRailsResult(
            ok=len(all_violations) == 0,
            violations=all_violations,
            text=text,
        )

    def __len__(self) -> int:
        return len(self._guards)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def check(text: str, *guards: Guardrail) -> GuardRailsResult:
    """
    Check *text* against the provided guardrails.

    Args:
        text: Text to check.
        *guards: Guardrail instances.

    Returns:
        :class:`GuardRailsResult`
    """
    return GuardRails(list(guards)).check(text)
