"""
agent-guard-rails: Composable output guardrails for agent responses.

Chain multiple rules (max length, banned phrases, required phrases, regex patterns)
and check any text output. Returns a GuardResult with pass/fail + violation messages.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps


@dataclass
class GuardResult:
    passed: bool
    violations: list[str] = field(default_factory=list)


class GuardRailViolationError(Exception):
    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__(f"Guard rail violations: {'; '.join(violations)}")


class GuardRail:
    """Base class for individual guardrail rules."""

    def check(self, text: str) -> list[str]:
        """Returns list of violation strings; empty list means pass."""
        raise NotImplementedError


class MaxLength(GuardRail):
    """Fails if the response exceeds max_chars characters."""

    def __init__(self, max_chars: int):
        self.max_chars = max_chars

    def check(self, text: str) -> list[str]:
        length = len(text)
        if length > self.max_chars:
            return [f"Response too long: {length} chars, max {self.max_chars}"]
        return []


class MinLength(GuardRail):
    """Fails if the response (stripped) is shorter than min_chars characters."""

    def __init__(self, min_chars: int):
        self.min_chars = min_chars

    def check(self, text: str) -> list[str]:
        length = len(text.strip())
        if length < self.min_chars:
            return [f"Response too short: {length} chars, min {self.min_chars}"]
        return []


class BannedPhrase(GuardRail):
    """Fails if any of the banned phrases are found in the text."""

    def __init__(self, *phrases: str, case_sensitive: bool = False):
        self.phrases = phrases
        self.case_sensitive = case_sensitive

    def check(self, text: str) -> list[str]:
        violations = []
        haystack = text if self.case_sensitive else text.lower()
        for phrase in self.phrases:
            needle = phrase if self.case_sensitive else phrase.lower()
            if needle in haystack:
                violations.append(f"Banned phrase found: '{phrase}'")
        return violations


class RequiredPhrase(GuardRail):
    """Fails if any of the required phrases are missing from the text."""

    def __init__(self, *phrases: str, case_sensitive: bool = False):
        self.phrases = phrases
        self.case_sensitive = case_sensitive

    def check(self, text: str) -> list[str]:
        violations = []
        haystack = text if self.case_sensitive else text.lower()
        for phrase in self.phrases:
            needle = phrase if self.case_sensitive else phrase.lower()
            if needle not in haystack:
                violations.append(f"Required phrase missing: '{phrase}'")
        return violations


class RegexMustMatch(GuardRail):
    """Fails if the text does NOT match the given regex pattern."""

    def __init__(self, pattern: str, message: str = ""):
        self.pattern = pattern
        self.message = message

    def check(self, text: str) -> list[str]:
        if not re.search(self.pattern, text):
            msg = self.message if self.message else f"Text must match pattern: {self.pattern}"
            return [msg]
        return []


class RegexMustNotMatch(GuardRail):
    """Fails if the text DOES match the given regex pattern."""

    def __init__(self, pattern: str, message: str = ""):
        self.pattern = pattern
        self.message = message

    def check(self, text: str) -> list[str]:
        if re.search(self.pattern, text):
            msg = self.message if self.message else f"Text must not match pattern: {self.pattern}"
            return [msg]
        return []


class GuardRails:
    """Chains multiple GuardRail rules and evaluates them against text output."""

    def __init__(self, rails: list[GuardRail] | None = None):
        self._rails: list[GuardRail] = list(rails) if rails is not None else []

    def add(self, rail: GuardRail) -> "GuardRails":
        """Append a rail; returns self for chaining."""
        self._rails.append(rail)
        return self

    def remove(self, index: int) -> "GuardRails":
        """Remove the rail at index; raises IndexError if out of bounds. Returns self."""
        if index < 0 or index >= len(self._rails):
            raise IndexError(f"Rail index {index} out of range (len={len(self._rails)})")
        del self._rails[index]
        return self

    def rails(self) -> list[GuardRail]:
        """Return a copy of the rail list."""
        return list(self._rails)

    def check(self, text: str) -> GuardResult:
        """Run all rails, collect ALL violations. passed=True only if no violations."""
        all_violations: list[str] = []
        for rail in self._rails:
            all_violations.extend(rail.check(text))
        return GuardResult(passed=len(all_violations) == 0, violations=all_violations)

    def enforce(self, text: str) -> str:
        """
        Run check(); raise GuardRailViolationError if any violations found.
        Returns text unchanged if all rails pass.
        """
        result = self.check(text)
        if not result.passed:
            raise GuardRailViolationError(result.violations)
        return text

    def guarded(self) -> Callable:
        """
        Decorator factory: wraps a function and calls enforce() on its string return value.

        Usage:
            @guardrails.guarded()
            def generate(...) -> str:
                ...
        """

        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                result = fn(*args, **kwargs)
                return self.enforce(result)

            return wrapper

        return decorator


__all__ = [
    "GuardResult",
    "GuardRailViolationError",
    "GuardRail",
    "MaxLength",
    "MinLength",
    "BannedPhrase",
    "RequiredPhrase",
    "RegexMustMatch",
    "RegexMustNotMatch",
    "GuardRails",
]
