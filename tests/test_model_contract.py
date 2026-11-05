"""Check safe context boundaries and exact causal action-probability alignment."""

import builtins
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from multi_agent_ssr.contracts import Generation, ModelConfig
from multi_agent_ssr.model import ContextLimitError, LocalPolicy, bound_messages


def encode_messages(messages):
    """One token per character makes context-removal cases explicit and deterministic."""
    return [1 for item in messages for _ in item.get("content", "")]


def test_context_eviction_preserves_tool_pairs_and_latest_state():
    messages = [
        {"role": "system", "content": "s"}, {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old", "tool_calls": [{"id": "old-id"}]},
        {"role": "tool", "content": "old-result", "tool_call_id": "old-id"},
        {"role": "assistant", "content": "new", "tool_calls": [{"id": "new-id"}]},
        {"role": "tool", "content": "new-result", "tool_call_id": "new-id"},
    ]
    retained, ids, removed = bound_messages(messages, encode_messages, 18)
    assert removed == 1
    assert retained == messages[:2] + messages[4:]
    assert len(ids) == 18
    assert len(messages) == 6  # The full transcript remains available for logging.


def test_overlong_required_context_fails_instead_of_slicing_tokens():
    with pytest.raises(ContextLimitError):
        bound_messages([{"role": "user", "content": "required task"}], encode_messages, 2)


def test_textual_tool_protocol_evicts_observations_with_their_action():
    messages = [
        {"role": "system", "content": "s"}, {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "Tool read result: old content"},
        {"role": "assistant", "content": "new"},
        {"role": "user", "content": "Tool read result: new content"},
    ]
    budget = len(encode_messages(messages[:2] + messages[4:]))
    retained, _, removed = bound_messages(messages, encode_messages, budget)
    assert retained == messages[:2] + messages[4:]
    assert removed == 1


def test_latest_observation_is_bounded_without_breaking_native_tool_ids():
    messages = [
        {"role": "system", "content": "instructions"}, {"role": "user", "content": "task"},
        {"role": "assistant", "content": "read", "tool_calls": [{"id": "call"}]},
        {"role": "tool", "content": "x" * 2000, "tool_call_id": "call"},
    ]
    retained, ids, removed = bound_messages(messages, encode_messages, 350)
    assert retained[:3] == messages[:3]
    assert retained[3]["tool_call_id"] == "call"
    assert "Observation shortened to fit context" in retained[3]["content"]
    assert len(ids) <= 350 and removed == 0
    assert len(messages[3]["content"]) == 2000


@pytest.mark.parametrize("messages", [
    [{"role": "tool", "content": "orphan"}],
    [{"role": "assistant", "tool_calls": [{"id": "x"}]},
     {"role": "tool", "tool_call_id": "y"}],
    [{"role": "assistant", "tool_calls": [{"id": "missing-result"}]}],
])
def test_orphan_or_unmatched_tool_results_are_rejected(messages):
    with pytest.raises(ValueError):
        bound_messages(messages, encode_messages, 100)


def test_importing_harness_backend_does_not_import_ml_libraries():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import multi_agent_ssr.model; "
         "assert 'torch' not in sys.modules; assert 'unsloth' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_unpinned_model_fails_before_any_download(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        LocalPolicy(ModelConfig(), tmp_path / "model.jsonl")


def test_statistics_downloads_are_disabled_before_unsloth_import(tmp_path, monkeypatch):
    """Stop at the import boundary; this test loads no ML library or model."""
    monkeypatch.delenv("UNSLOTH_DISABLE_STATISTICS", raising=False)
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising=False)
    original_import = builtins.__import__

    class ImportReached(Exception):
        pass

    def intercept_import(name, *args, **kwargs):
        if name == "unsloth":
            assert os.environ.get("UNSLOTH_DISABLE_STATISTICS") == "1"
            raise ImportReached
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", intercept_import)
    with pytest.raises(ImportReached):
        LocalPolicy(ModelConfig(revision="fixture-pin"), tmp_path / "model.jsonl")


def test_action_logprobs_align_first_completion_and_exclude_observations():
    """A positional toy model exposes off-by-one and prompt-loss mistakes."""
    torch = pytest.importorskip("torch")

    class PositionalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.arange(35, dtype=torch.float32).reshape(5, 7) / 10)

        def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
            assert logits_to_keep == 3
            assert not use_cache
            return SimpleNamespace(logits=self.logits[-logits_to_keep:].unsqueeze(0))

    policy = LocalPolicy.__new__(LocalPolicy)
    policy.torch = torch
    policy.config = ModelConfig(context_tokens=8)
    policy.device = torch.device("cpu")
    policy.lock = threading.RLock()
    policy.model = PositionalModel()
    policy._backend = SimpleNamespace(for_training=lambda model: model.train())
    generation = Generation("", [1, 2, 3], [4, 5], [0.0, 0.0])
    actual = policy.action_logprobs(generation)
    expected = torch.stack([policy.model.logits[2].log_softmax(0)[4],
                            policy.model.logits[3].log_softmax(0)[5]])
    assert torch.allclose(actual, expected)
    (-actual.sum()).backward()
    assert policy.model.logits.grad[:2].abs().sum() == 0
    assert policy.model.logits.grad[2:4].abs().sum() > 0
    assert policy.model.logits.grad[4].abs().sum() == 0


def test_action_scoring_uses_bf16_forward_and_float32_normalization():
    """Exercise a real matrix product so autocast cannot be a no-op mock."""
    torch = pytest.importorskip("torch")

    class MatmulModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(35, dtype=torch.float32).reshape(5, 7) / 11)

        def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
            output = torch.eye(5) @ self.weight
            assert torch.is_autocast_enabled("cpu")
            assert output.dtype == torch.bfloat16
            return SimpleNamespace(logits=output[-logits_to_keep:].unsqueeze(0))

    policy = LocalPolicy.__new__(LocalPolicy)
    policy.torch = torch
    policy.config = ModelConfig(context_tokens=8)
    policy.device = "cpu"
    policy.lock = threading.RLock()
    policy.model = MatmulModel()
    policy._backend = SimpleNamespace(for_training=lambda model: model.train())
    generation = Generation("", [1, 2, 3], [4, 5], [0.0, 0.0])
    actual = policy.action_logprobs(generation)
    expected = policy.model.weight.detach().to(torch.bfloat16).float().log_softmax(-1)
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected[[2, 3], [4, 5]], atol=1e-6)
    assert not torch.is_autocast_enabled("cpu")
    (-actual.sum()).backward()
    assert torch.isfinite(policy.model.weight.grad).all()
    assert policy.model.weight.grad[2:4].abs().sum() > 0
    assert policy.model.weight.grad[[0, 1, 4]].abs().sum() == 0


@pytest.mark.parametrize("initial_training", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_behavior_replay_measures_before_forcing_and_restores_state(tmp_path, initial_training, fail):
    """Forced token probability is one; replay must instead report raw probabilities."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class ReplayModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32))
            self.generation_config = SimpleNamespace(eos_token_id=5, bos_token_id=1)

        def generate(self, input_ids, attention_mask, generation_config, logits_processor, stopping_criteria):
            assert not self.training and not torch.is_grad_enabled()
            assert generation_config.max_new_tokens == 4  # Original capacity, not the saved length.
            assert generation_config.temperature == 1 and generation_config.top_k == 0
            assert generation_config.use_cache
            result = input_ids
            for step, token in enumerate((1, 3)):
                scores = self.weight.unsqueeze(0) + step * torch.arange(6)
                raw = scores.clone()
                forced = logits_processor(result, scores)
                assert torch.equal(scores, raw)  # The measurement input is not overwritten.
                assert forced.softmax(-1)[0, token] == 1
                sampled = torch.multinomial(forced.softmax(-1), num_samples=1)
                result = torch.cat((result, sampled), dim=1)
                assert bool(stopping_criteria(result, None).all()) == (step == 1)
                if fail:
                    raise RuntimeError("fixture replay failure")
            return result

    policy = LocalPolicy.__new__(LocalPolicy)
    policy.torch = torch
    policy.config = ModelConfig(context_tokens=32, action_tokens=4)
    policy.device = torch.device("cpu")
    policy.lock = threading.RLock()
    policy._log_lock = threading.RLock()
    policy.log_path = tmp_path / "replay.jsonl"
    policy.version = 7
    policy.model = ReplayModel().train(initial_training)
    policy.tokenizer = SimpleNamespace(pad_token_id=0)
    policy._backend = SimpleNamespace(for_inference=lambda model: model.eval(),
                                      for_training=lambda model: model.train())
    generation = Generation("fixture", [1, 2], [1, 3], [-5.0, -6.0], policy_version=7)
    old = generation.old_logprobs.copy()
    weight = policy.model.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    if fail:
        with pytest.raises(RuntimeError, match="fixture replay failure"):
            policy.replay_behavior_logprobs(generation, max_new_tokens=4)
    else:
        actual = policy.replay_behavior_logprobs(generation, max_new_tokens=4)
        expected = [torch.arange(6, dtype=torch.float32).log_softmax(0)[1],
                    (2 * torch.arange(6, dtype=torch.float32)).log_softmax(0)[3]]
        assert actual == pytest.approx([float(value) for value in expected])
        event = json.loads(policy.log_path.read_text().splitlines()[-1])
        assert event["event"] == "behavior_replay_complete"
        assert event["completion_ids"] == generation.completion_ids
        assert event["prompt_ids"] == generation.prompt_ids
    assert generation.old_logprobs == old
    assert policy.model.training == initial_training
    assert torch.equal(policy.model.weight, weight)
    assert policy.model.weight.grad is None
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("long_observation", [False, True])
def test_generation_records_the_actual_sampled_token_probability(tmp_path, monkeypatch, long_observation):
    """Verify the final/EOS-step recorder path without loading any model weights."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token-never-log-this")

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return json.dumps(messages) if long_observation else "prompt"

        def encode(self, text, **kwargs):
            return [1] * len(text) if long_observation else [1, 2]

        def decode(self, ids, **kwargs):
            return " ".join(str(int(value)) for value in ids)

    class GeneratingModel:
        generation_config = SimpleNamespace(eos_token_id=5, bos_token_id=1)

        def generate(self, input_ids, attention_mask, generation_config, logits_processor, streamer):
            assert generation_config.temperature == 1
            assert generation_config.top_k == 0
            streamer.put(input_ids.cpu())
            result = input_ids
            for token in (4, 5):
                scores = torch.arange(6, dtype=torch.float32).unsqueeze(0)
                logits_processor(result, scores)
                result = torch.cat([result, torch.tensor([[token]])], dim=1)
                streamer.put(torch.tensor([token]))
            streamer.end()
            return result

    policy = LocalPolicy.__new__(LocalPolicy)
    policy.config = ModelConfig(context_tokens=512 if long_observation else 32, action_tokens=4)
    policy.log_path = tmp_path / "events.jsonl"
    policy.lock = threading.RLock()
    policy._log_lock = threading.RLock()
    policy.version = 7
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.processor = policy.tokenizer = Tokenizer()
    policy.model = GeneratingModel()
    policy._backend = SimpleNamespace(for_inference=lambda model: None,
                                      for_training=lambda model: None)
    messages = [{"role": "user", "content": "test"}]
    if long_observation:
        messages += [{"role": "assistant", "content": "read"},
                     {"role": "user", "content": "Tool read result: " + "x" * 2000}]
    result = policy.generate(messages, max_new_tokens=4)
    expected = torch.arange(6, dtype=torch.float32).log_softmax(0)[[4, 5]].tolist()
    assert result.old_logprobs == pytest.approx(expected)
    if not long_observation:
        assert result.prompt_ids == [1, 2]
    assert result.completion_ids == [4, 5]
    assert result.policy_version == 7
    content = policy.log_path.read_text(encoding="utf-8")
    assert "not-a-real-token-never-log-this" not in content
    events = [json.loads(line) for line in content.splitlines()]
    effective = next(item for item in events if item["event"] == "prompt_effective")
    requested = next(item for item in events if item["event"] == "prompt_requested")
    assert effective["prompt_ids"] == result.prompt_ids
    assert requested["messages"] == messages
    if long_observation:
        assert "Observation shortened to fit context" in effective["messages"][-1]["content"]
        assert len(result.prompt_ids) <= policy.config.context_tokens - 4
        assert result.prompt_ids == policy._encode_messages(effective["messages"], None)
    assert any(item["event"] == "output_chunk" for item in events)
    assert events[-1]["event"] == "generation_complete"
