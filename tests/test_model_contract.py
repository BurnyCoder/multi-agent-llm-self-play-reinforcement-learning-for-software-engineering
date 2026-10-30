"""Check safe context boundaries and exact causal action-probability alignment."""

import json
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


def test_generation_records_the_actual_sampled_token_probability(tmp_path, monkeypatch):
    """Verify the final/EOS-step recorder path without loading any model weights."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token-never-log-this")

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "prompt"

        def encode(self, text, **kwargs):
            return [1, 2]

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
    policy.config = ModelConfig(context_tokens=32, action_tokens=4)
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
    result = policy.generate([{"role": "user", "content": "test"}], max_new_tokens=4)
    expected = torch.arange(6, dtype=torch.float32).log_softmax(0)[[4, 5]].tolist()
    assert result.old_logprobs == pytest.approx(expected)
    assert result.prompt_ids == [1, 2]
    assert result.completion_ids == [4, 5]
    assert result.policy_version == 7
    content = policy.log_path.read_text(encoding="utf-8")
    assert "not-a-real-token-never-log-this" not in content
    events = [json.loads(line) for line in content.splitlines()]
    assert any(item["event"] == "output_chunk" for item in events)
    assert events[-1]["event"] == "generation_complete"
