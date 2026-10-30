"""Check reward equations, grouped credit assignment and actual optimizer effects."""

import math
import threading
from types import SimpleNamespace

import pytest

from multi_agent_ssr.contracts import ActionTrace, Episode, Generation
from multi_agent_ssr.learning import (
    clipped_token_loss,
    compute_advantages,
    injection_reward,
    update_policy,
)


def make_episode(identifier, reward, tokens=(1,), *, team="solver", task="same-task",
                 version=0, completed=True):
    """Build independent action records; old probabilities match a uniform policy."""
    generation = Generation("", [1], list(tokens), [-math.log(2)] * len(tokens), version)
    action = ActionTrace(identifier + "/call", identifier, None, "general", generation)
    return Episode(identifier, task, team, [action], reward=reward, completed=completed)


@pytest.mark.parametrize("valid,rate,expected", [
    (False, None, -1.0), (True, 0.0, -0.8), (True, 1.0, -0.8),
    (True, 0.25, 0.55), (True, 0.5, 0.1), (True, 0.75, -0.35),
])
def test_ssr_injection_reward(valid, rate, expected):
    """Separate consistency failure from valid but degenerate difficulty."""
    assert injection_reward(valid, rate) == pytest.approx(expected)


@pytest.mark.parametrize("rate", [None, float("nan"), -0.1, 1.1])
def test_valid_bug_requires_a_real_solve_rate(rate):
    with pytest.raises(ValueError):
        injection_reward(True, rate)


def test_advantages_weight_unique_root_and_child_actions():
    """Children count once even when their answer is later repeated in a prompt."""
    success = make_episode("success", 1.0)
    child = Generation("answer", [1, 1, 1, 1], [1, 1], [-math.log(2)] * 2)
    success.actions.append(ActionTrace("child/call", "child", "success", "explore", child))
    failure = make_episode("failure", -1.0, tokens=(0,))
    advantages = compute_advantages([success, failure])
    assert advantages == pytest.approx({"success": 0.5, "failure": -1.5})
    assert sum(ep.token_count * advantages[ep.episode_id] for ep in [success, failure]) == 0


@pytest.mark.parametrize("alteration", ["team", "task", "incomplete", "missing-reward", "duplicate"])
def test_invalid_comparison_groups_are_rejected(alteration):
    first, second = make_episode("first", 1), make_episode("second", -1)
    if alteration == "team":
        second.team = "injector"
    elif alteration == "task":
        second.task_id = "different-bug"
    elif alteration == "incomplete":
        second.completed = False
    elif alteration == "missing-reward":
        second.reward = None
    else:
        second.episode_id = first.episode_id
    with pytest.raises(ValueError):
        compute_advantages([first, second])


def test_clipping_has_correct_positive_and_negative_gradient_regions():
    """Exercise both asymmetric PPO clipping boundaries, not just finite loss."""
    torch = pytest.importorskip("torch")
    for advantage, expected, zero_index in [(1.0, -2.75 / 8, 2), (-1.0, 3.3 / 8, 0)]:
        new = torch.tensor([math.log(0.5), 0.0, math.log(1.5)], requires_grad=True)
        old = torch.zeros(3, requires_grad=True)
        loss = clipped_token_loss(new, old, advantage, capacity=8)
        assert loss.item() == pytest.approx(expected)
        loss.backward()
        assert new.grad[zero_index] == 0
        assert old.grad is None
        assert torch.isfinite(new.grad).all()


def test_loss_uses_fixed_capacity_not_observed_action_length():
    torch = pytest.importorskip("torch")
    short = clipped_token_loss(torch.zeros(1), torch.zeros(1), 1.0, 8)
    long = clipped_token_loss(torch.zeros(3), torch.zeros(3), 1.0, 8)
    assert long.item() == pytest.approx(3 * short.item())


class TinyPolicy:
    """A two-action differentiable policy for proof-oriented CPU update tests."""

    def __init__(self, torch):
        self.torch = torch
        self.weights = torch.nn.Parameter(torch.zeros(2))
        self.lock = threading.RLock()
        self.version = 0
        self.config = SimpleNamespace(learning_rate=0.05)
        self.scored = []

    def trainable_parameters(self):
        return [self.weights]

    def action_logprobs(self, generation):
        self.scored.append(tuple(generation.completion_ids))
        return self.weights.log_softmax(0)[generation.completion_ids]

    def adapter_fingerprint(self):
        return str(self.weights.detach().tolist())


def test_expired_update_never_starts_an_action_or_optimizer_step():
    torch = pytest.importorskip("torch")
    policy = TinyPolicy(torch)
    optimizer = torch.optim.SGD(policy.trainable_parameters(), lr=0.1)
    episodes = [make_episode("success", 1, (1,)), make_episode("failure", -1, (0,))]
    with pytest.raises(TimeoutError, match="evaluation reserve"):
        update_policy(policy, [episodes], optimizer, capacity=8, deadline=0)
    assert not policy.scored
    assert policy.version == 0


def test_zero_action_group_is_scored_but_never_trained():
    torch = pytest.importorskip("torch")
    policy = TinyPolicy(torch)
    episodes = [Episode("empty-a", "same", "solver", reward=-1, completed=True),
                Episode("empty-b", "same", "solver", reward=-1, completed=True)]
    optimizer = torch.optim.SGD(policy.trainable_parameters(), lr=0.1)
    result = update_policy(policy, [episodes], optimizer, capacity=8)
    assert result["no_action_groups"] == 1
    assert result["optimizer_steps"] == 0
    assert not policy.scored


def test_shared_policy_updates_and_reuses_behavior_logprobs_across_passes():
    torch = pytest.importorskip("torch")
    policy = TinyPolicy(torch)
    optimizer = torch.optim.SGD(policy.trainable_parameters(), lr=0.1)
    episodes = [make_episode("success", 1, (1,)), make_episode("failure", -1, (0,))]
    original_logps = [episode.actions[0].generation.old_logprobs[:] for episode in episodes]
    metrics = update_policy(policy, [episodes], optimizer, capacity=8, epochs=2)
    assert policy.weights.softmax(0)[1] > 0.5
    assert policy.version == metrics["policy_version"] == 2
    assert metrics["optimizer_steps"] == 2
    assert policy.scored == [(1,), (0,), (1,), (0,)]
    assert [episode.actions[0].generation.old_logprobs for episode in episodes] == original_logps
    assert all(math.isfinite(item["loss"]) for item in metrics["passes"])


def test_zero_advantage_group_does_not_fabricate_an_optimizer_step():
    torch = pytest.importorskip("torch")
    policy = TinyPolicy(torch)
    before = policy.adapter_fingerprint()
    optimizer = torch.optim.SGD(policy.trainable_parameters(), lr=0.1)
    group = [make_episode("a", -1), make_episode("b", -1)]
    result = update_policy(policy, [group], optimizer, capacity=8)
    assert result["optimizer_steps"] == policy.version == 0
    assert policy.adapter_fingerprint() == before
    assert not policy.scored


@pytest.mark.parametrize("problem", ["stale", "duplicate-call", "capacity", "logprobs"])
def test_update_rejects_incoherent_rollouts_before_mutation(problem):
    torch = pytest.importorskip("torch")
    policy = TinyPolicy(torch)
    optimizer = torch.optim.SGD(policy.trainable_parameters(), lr=0.1)
    group = [make_episode("a", 1), make_episode("b", -1, (0,))]
    capacity = 8
    if problem == "stale":
        group[1].actions[0].generation.policy_version = 1
    elif problem == "duplicate-call":
        group[1].actions[0].call_id = group[0].actions[0].call_id
    elif problem == "capacity":
        capacity = 1
    else:
        group[1].actions[0].generation.old_logprobs = []
    before = policy.adapter_fingerprint()
    with pytest.raises(ValueError):
        update_policy(policy, [group], optimizer, capacity)
    assert policy.adapter_fingerprint() == before
    assert policy.version == 0
