"""Failed likelihood checks must retain evidence and never reach optimization."""

import json
import math
import threading

import pytest

from multi_agent_ssr.contracts import Generation, ModelConfig


class StressForwardReached(Exception):
    """Stop a passing diagnostic before synthetic forward/backward execution."""


@pytest.mark.parametrize("scoring_gap,replay_gap,error", [
    (0.2, 0.0, StressForwardReached),
    (-0.2, 0.0, StressForwardReached),
    (math.log(0.8), 0.0, StressForwardReached),
    (math.log(1.25), 0.0, StressForwardReached),
    (0.23, 0.0, RuntimeError),
    (-0.23, 0.0, RuntimeError),
    (1000.0, 0.0, RuntimeError),
    (-1000.0, 0.0, RuntimeError),
    (0.2, 0.1, RuntimeError),
    (float("nan"), 0.0, FloatingPointError),
    (float("inf"), 0.0, FloatingPointError),
    (0.2, float("nan"), FloatingPointError),
])
def test_preflight_records_likelihood_guards_before_optimization(
        tmp_path, monkeypatch, scoring_gap, replay_gap, error):
    """Distinguish cached replay correctness from finite BF16 scorer drift."""
    from multi_agent_ssr import learning, model

    torch = pytest.importorskip("torch")
    original = [-1.0, -2.0]
    scoring = [original[0] + scoring_gap, original[1]]
    replayed = [original[0] + replay_gap, original[1]]
    zero_grad_calls = []
    logged = []

    class DiagnosticOptimizer:
        def zero_grad(self, **kwargs):
            zero_grad_calls.append(kwargs)

    class DiagnosticPolicy:
        def __init__(self, *args):
            self.torch, self.lock = torch, threading.RLock()

        def adapter_fingerprint(self):
            return "unchanged-adapter"

        def generate(self, *args, **kwargs):
            return Generation("fixture", [1], [2, 3], original.copy())

        def action_logprobs(self, generation):
            if generation.text == "preflight synthetic tensor shape only":
                raise StressForwardReached
            assert generation.old_logprobs == original
            return torch.tensor(scoring, dtype=torch.float64)

        def replay_behavior_logprobs(self, generation, *, max_new_tokens):
            assert max_new_tokens == 256 and generation.old_logprobs == original
            return replayed

        def log_event(self, event, **fields):
            assert event == "likelihood_checked"
            logged.append(fields)

    monkeypatch.setattr(model, "LocalPolicy", DiagnosticPolicy)
    # There is no optimizer step method, and a passing guard stops at stress.
    monkeypatch.setattr(learning, "build_optimizer", lambda policy: DiagnosticOptimizer())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    with pytest.raises(error):
        learning.run_preflight(ModelConfig(), tmp_path)
    evidence = json.loads((tmp_path / "likelihood.json").read_text())
    assert evidence == logged[0]
    assert evidence["sampled_logprobs"] == original
    assert len(evidence["scoring_logprobs"]) == len(evidence["replayed_logprobs"]) == len(original)
    assert evidence["difference_limit"] is None
    assert evidence["replay_limit"] == 0.05
    assert evidence["scoring_guard"] == {
        "kind": "preupdate_ratios_inside_clip_interval", "ratio_lower": 0.8, "ratio_upper": 1.25,
        "log_ratio_lower": math.log(0.8), "log_ratio_upper": math.log(1.25),
    }
    assert evidence["finite"] == (math.isfinite(scoring_gap) and math.isfinite(replay_gap))
    if math.isfinite(scoring_gap):
        assert evidence["scoring_max_abs_difference"] == pytest.approx(abs(scoring_gap))
        assert evidence["scoring_log_ratio_min"] == pytest.approx(min(0.0, scoring_gap))
        assert evidence["scoring_log_ratio_max"] == pytest.approx(max(0.0, scoring_gap))
        assert evidence["scoring_log_ratio_mean"] == pytest.approx(scoring_gap / 2)
        assert evidence["scoring_ratio_min"] == pytest.approx(math.exp(min(0.0, scoring_gap)))
        expected_max = math.exp(max(0.0, scoring_gap)) if scoring_gap < 710 else "inf"
        if expected_max == "inf":
            assert evidence["scoring_ratio_max"] == "inf"
        else:
            assert evidence["scoring_ratio_max"] == pytest.approx(expected_max)
        clipped = int(scoring_gap < math.log(0.8) or scoring_gap > math.log(1.25))
        assert evidence["scoring_clipped_tokens"] == clipped
        assert evidence["scoring_clipped_fraction"] == clipped / len(original)
    else:
        assert evidence["scoring_max_abs_difference"] is None
        assert evidence["scoring_log_ratio_min"] is None
        assert evidence["scoring_clipped_fraction"] is None
    if math.isfinite(replay_gap):
        assert evidence["replay_max_abs_difference"] == pytest.approx(abs(replay_gap))
    else:
        assert evidence["replay_max_abs_difference"] is None
    assert len(zero_grad_calls) == (1 if error is StressForwardReached else 0)
    assert not (tmp_path / "checkpoint").exists()
