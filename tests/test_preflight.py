"""Failed likelihood checks must retain evidence and never reach optimization."""

import json
import threading

import pytest

from multi_agent_ssr.contracts import Generation, ModelConfig


@pytest.mark.parametrize("replay_error", [False, True])
def test_preflight_records_both_likelihood_checks_before_rejecting(tmp_path, monkeypatch, replay_error):
    from multi_agent_ssr import learning, model

    torch = pytest.importorskip("torch")
    original = [-1.0, -2.0]

    class DiagnosticPolicy:
        def __init__(self, *args):
            self.torch, self.lock = torch, threading.RLock()

        def adapter_fingerprint(self):
            return "unchanged-adapter"

        def generate(self, *args, **kwargs):
            return Generation("fixture", [1], [2, 3], original.copy())

        def action_logprobs(self, generation):
            assert generation.old_logprobs == original
            return torch.tensor([-1.2, -2.0])

        def replay_behavior_logprobs(self, generation, *, max_new_tokens):
            assert max_new_tokens == 256 and generation.old_logprobs == original
            return [-1.1, -2.0] if replay_error else original.copy()

        def log_event(self, event, **fields):
            assert event == "likelihood_checked"

    monkeypatch.setattr(model, "LocalPolicy", DiagnosticPolicy)
    # No optimizer methods exist: reaching an update would fail this test.
    monkeypatch.setattr(learning, "build_optimizer", lambda policy: object())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    with pytest.raises(RuntimeError, match="log probabilities disagree"):
        learning.run_preflight(ModelConfig(), tmp_path)
    evidence = json.loads((tmp_path / "likelihood.json").read_text())
    assert evidence["sampled_logprobs"] == original
    assert evidence["scoring_max_abs_difference"] == pytest.approx(0.2)
    assert evidence["replay_max_abs_difference"] == pytest.approx(0.1 if replay_error else 0.0)
    assert evidence["difference_limit"] == 0.05
    assert not (tmp_path / "checkpoint").exists()
