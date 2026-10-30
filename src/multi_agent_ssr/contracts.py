"""Shared records at the boundaries of rollout, validation, learning and reporting.

Exact sampled token records are authoritative: independently delegated contexts
must never be flattened into a fictitious conversation for policy gradients.
Source: https://arxiv.org/html/2510.02387v1#A3
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class ModelConfig:
    """One shared policy; these are preregistered defaults, not measured results."""

    model_id: str = "Qwen/Qwen3.5-2B"
    revision: str = ""
    context_tokens: int = 2048
    action_tokens: int = 256
    lora_rank: int = 16
    lora_alpha: int = 16
    learning_rate: float = 5e-6
    seed: int = 0


@dataclass
class Limits:
    """Team-wide caps include children, including those still running at finish."""

    generated_tokens: int = 4096
    tool_calls: int = 32
    child_sessions: int = 2
    child_depth: int = 1
    command_seconds: int = 90
    episode_seconds: int = 900


@dataclass
class Generation:
    """The precise model input and sampled continuation for one inference call."""

    text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    old_logprobs: list[float]
    policy_version: int = 0
    elapsed_seconds: float = 0.0


class Policy(Protocol):
    """Backends serialize shared GPU access while agent/tool threads can overlap."""

    version: int

    def generate(self, messages: list[dict[str, Any]], *, max_new_tokens: int,
                 tools: list[dict[str, Any]] | None = None) -> Generation: ...


@dataclass
class ActionTrace:
    """A sampled call belongs to exactly one session and one team episode."""

    call_id: str
    session_id: str
    parent_session_id: str | None
    role: str
    generation: Generation
    timestamp: str = ""


@dataclass
class Episode:
    """Root and all child actions share the team's terminal return."""

    episode_id: str
    task_id: str
    team: str
    actions: list[ActionTrace] = field(default_factory=list)
    reward: float | None = None
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    completed: bool = False
    elapsed_seconds: float = 0.0

    @property
    def token_count(self) -> int:
        """Count sampled tokens once, excluding copies used as observations."""
        return sum(len(action.generation.completion_ids) for action in self.actions)

    def to_dict(self) -> dict[str, Any]:
        """Use a stable JSON-compatible representation at persistence boundaries."""
        return asdict(self)


@dataclass
class RepoSpec:
    """A pinned, licensed source snapshot and its runnable environment."""

    name: str
    url: str
    commit: str
    image: str
    source_path: str
    split: str
    source_roots: list[str]
    test_roots: list[str]
    license: str = ""
    image_digest: str = ""


@dataclass
class BugArtifact:
    """SSR's bug/test separation, with a trusted Python runner adaptation."""

    task_id: str
    repo_name: str
    bug_patch: str
    weaken_patch: str
    test_files: list[str]
    test_targets: list[str]
    order: int = 1
    parent_task_id: str | None = None


@dataclass
class ValidationResult:
    """Only executable checks establish artifact validity, not agent claims."""

    valid: bool
    reason: str
    pass_to_pass: list[str] = field(default_factory=list)
    fail_to_pass: list[str] = field(default_factory=list)
    clean_results: dict[str, str] = field(default_factory=dict)
    bug_results: dict[str, str] = field(default_factory=dict)
    weak_results: dict[str, str] = field(default_factory=dict)
    buggy_tree: str = ""


def episode_from_dict(data: dict[str, Any]) -> Episode:
    """Restore exact tokens and parent identities for checkpoint/replay audits."""
    data = dict(data)
    data["actions"] = [ActionTrace(**{**action, "generation": Generation(**action["generation"])})
                       for action in data.get("actions", [])]
    return Episode(**data)
