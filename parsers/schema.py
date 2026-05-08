from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Step:
    agent: str
    content: str
    kind: Literal["message", "tool_call", "tool_result", "system"]
    metadata: dict = field(default_factory=dict)


@dataclass
class Trace:
    steps: list[Step]
    metadata: dict = field(default_factory=dict)
