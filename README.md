# agent-guard-rails

Composable output guardrails for LLM responses. Zero dependencies.

```python
from agent_guard_rails import GuardRails, MaxLength, ForbiddenPhrase, RequiredPhrase

rails = GuardRails([
    MaxLength(2000),
    ForbiddenPhrase("I don't know"),
    RequiredPhrase("DONE"),
])

result = rails.check(response_text)
if not result.ok:
    for v in result.violations:
        print(v)
```

## Install

```bash
pip install agent-guard-rails
```
