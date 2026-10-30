"""SSR rewards and the CWM objective, factored over independent team contexts.

Rewards: https://arxiv.org/html/2512.18552v3#S2.SS3 .
Masked, length-weighted clipped objective: https://arxiv.org/html/2510.02387v1#A3 .
Assigning one team's return to its root/child action tree is this project's
extension; it is not a convergence guarantee or an exact reproduction of SSR.
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import Episode, Generation, ModelConfig


def injection_reward(valid: bool, solve_rate: float | None, alpha: float = 0.8) -> float:
    """Implement SSR equation 1, including the distinct invalid-bug penalty."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if not valid:
        return -1.0
    if solve_rate is None or not math.isfinite(solve_rate) or not 0 <= solve_rate <= 1:
        raise ValueError("A valid bug requires a finite solve rate in [0, 1]")
    if solve_rate in (0.0, 1.0):
        return -alpha
    return 1.0 - (1.0 + alpha) * solve_rate


def compute_advantages(episodes: list[Episode]) -> dict[str, float]:
    """Center team returns by their unique assistant-token-weighted group mean."""
    if len(episodes) < 2:
        raise ValueError("A complete comparison group requires at least two episodes")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("Episode IDs must be unique within a group")
    if len({(episode.team, episode.task_id) for episode in episodes}) != 1:
        raise ValueError("A reward group must share one team and identical task_id")
    if any(not episode.completed or episode.reward is None or not math.isfinite(episode.reward)
           for episode in episodes):
        raise ValueError("Only completed, finite-reward episodes can form a training group")
    total = sum(episode.token_count for episode in episodes)
    if total == 0:
        # The weighted mean is undefined when no action was sampled. Returning
        # zero contribution preserves its scored outcomes without inventing
        # gradients; update_policy reports this non-trainable group explicitly.
        return {episode.episode_id: 0.0 for episode in episodes}
    mean = math.fsum(float(episode.reward) * episode.token_count for episode in episodes) / total
    return {episode.episode_id: float(episode.reward) - mean for episode in episodes}


def clipped_token_loss(new_logps: Any, old_logps: Any, advantage: float,
                       capacity: int) -> Any:
    """Return the negative PPO surrogate divided by a fixed token capacity.

    Only actual sampled action tokens are passed here. No per-sequence mean,
    per-agent mean, standard-deviation normalization, or KL term is introduced.
    """
    import torch

    if capacity < 1 or not math.isfinite(advantage):
        raise ValueError("Loss capacity must be positive and advantage finite")
    old = torch.as_tensor(old_logps, dtype=new_logps.dtype, device=new_logps.device).detach()
    if new_logps.ndim != 1 or new_logps.shape != old.shape or new_logps.numel() == 0:
        raise ValueError("Old and new log probabilities must match one nonempty action")
    if not torch.isfinite(new_logps).all() or not torch.isfinite(old).all():
        raise FloatingPointError("Nonfinite token log probability")
    ratio = (new_logps - old).exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("Nonfinite importance ratio")
    return -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.25) * advantage).sum() / capacity


def build_optimizer(policy: Any) -> Any:
    """Optimize the shared small adapter with the standard Torch AdamW optimizer."""
    import torch

    optimizer = torch.optim.AdamW(policy.trainable_parameters(),
                                  lr=policy.config.learning_rate,
                                  betas=(0.9, 0.99), weight_decay=0.0)
    pending = getattr(policy, "_pending_optimizer_state", None)
    if pending is not None:
        optimizer.load_state_dict(pending)
        policy._pending_optimizer_state = None
    return optimizer


def update_policy(policy: Any, groups: list[list[Episode]], optimizer: Any,
                  capacity: int, epochs: int = 2, deadline: float | None = None) -> dict[str, Any]:
    """Accumulate one action at a time, taking one optimizer step per pass.

    All rollout groups are collected under one unchanged shared policy before
    this call. The second pass retains the original behavior log probabilities.
    The caller sets capacity to the preregistered *maximum* batch token budget,
    not the observed trajectory length. Empty-advantage batches do not update.
    """
    import torch

    if epochs < 1 or capacity < 1 or not groups:
        raise ValueError("Training requires groups, positive capacity and positive epochs")
    started = time.monotonic()
    with policy.lock:
        initial_version = policy.version
        advantage_by_id: dict[str, float] = {}
        episodes: list[Episode] = []
        call_ids: set[str] = set()
        for group in groups:
            advantages = compute_advantages(group)
            if set(advantage_by_id).intersection(advantages):
                raise ValueError("An episode was duplicated across reward groups")
            advantage_by_id.update(advantages)
            episodes.extend(group)
            for episode in group:
                if episode.team not in {"injector", "solver"}:
                    raise ValueError("Training team must be 'injector' or 'solver'")
                for action in episode.actions:
                    generation = action.generation
                    if action.call_id in call_ids:
                        raise ValueError("An agent action was counted more than once")
                    call_ids.add(action.call_id)
                    if generation.policy_version != initial_version:
                        raise ValueError("Rollout groups must match the current behavior policy version")
                    if len(generation.completion_ids) != len(generation.old_logprobs):
                        raise ValueError("Stored behavior log probabilities do not align with action tokens")
                    if not generation.completion_ids or not generation.prompt_ids:
                        raise ValueError("Each sampled action needs a nonempty prompt and completion")
                    if not all(math.isfinite(value) for value in generation.old_logprobs):
                        raise FloatingPointError("Stored behavior probabilities are not finite")
        sampled_tokens = sum(episode.token_count for episode in episodes)
        if sampled_tokens > capacity:
            raise ValueError("Observed action tokens exceed the fixed batch token capacity")
        active = [episode for episode in episodes if advantage_by_id[episode.episode_id] != 0.0]
        trained_tokens = sum(episode.token_count for episode in active)
        metrics: dict[str, Any] = {
            "behavior_policy_version": initial_version, "groups": len(groups),
            "episodes": len(episodes), "sampled_tokens": sampled_tokens,
            "trained_tokens_per_pass": trained_tokens, "capacity": capacity,
            "zero_advantage_episodes": len(episodes) - len(active),
            "no_action_groups": sum(not any(ep.token_count for ep in group) for group in groups),
            "passes": [], "optimizer_steps": 0,
        }
        for epoch in range(epochs if trained_tokens else 0):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            clipped_count = 0
            try:
                for episode in active:
                    advantage = advantage_by_id[episode.episode_id]
                    for action in episode.actions:
                        if deadline is not None and time.time() >= deadline:
                            raise TimeoutError("Optimization reached the evaluation reserve; restore the last completed checkpoint")
                        logps = policy.action_logprobs(action.generation)
                        old = torch.tensor(action.generation.old_logprobs,
                                           device=logps.device, dtype=logps.dtype)
                        loss = clipped_token_loss(logps, old, advantage, capacity)
                        with torch.no_grad():
                            ratios = (logps - old).exp()
                            clipped_count += int(((ratios < 0.8) | (ratios > 1.25)).sum().item())
                        total_loss += float(loss.detach().item())
                        loss.backward()
                        del loss, logps, old
                parameters = policy.trainable_parameters()
                if not any(parameter.grad is not None for parameter in parameters):
                    raise RuntimeError("No adapter parameter received a gradient")
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in parameters):
                    raise FloatingPointError("Nonfinite adapter gradient; optimizer step was skipped")
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0,
                                                          error_if_nonfinite=True)
                optimizer.step()
                policy.version += 1
                metrics["optimizer_steps"] += 1
                metrics["passes"].append({"epoch": epoch, "loss": total_loss,
                                           "gradient_norm_before_clipping": float(grad_norm),
                                           "clip_fraction": clipped_count / trained_tokens})
            except BaseException:
                optimizer.zero_grad(set_to_none=True)
                raise
        optimizer.zero_grad(set_to_none=True)
        metrics.update(policy_version=policy.version, elapsed_seconds=time.monotonic() - started,
                       adapter_fingerprint=policy.adapter_fingerprint())
        if hasattr(policy, "log_event"):
            policy.log_event("policy_update", **metrics)
        return metrics


def run_preflight(config: ModelConfig, run_dir: Path) -> dict[str, Any]:
    """Measure a real local forward/backward/save/reload before the pilot.

    This explicitly loads the model; the CLI must invoke it only after the
    preregistration push. The synthetic update checkpoint is labeled preflight
    and must never initialize the actual research run.
    """
    from .model import LocalPolicy

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    policy = LocalPolicy(config, run_dir / "model.jsonl")
    torch = policy.torch
    optimizer = build_optimizer(policy)
    with policy.lock:
        torch.cuda.reset_peak_memory_stats()
        initial = policy.adapter_fingerprint()
        generation = policy.generate([
            {"role": "system", "content": "You are a Python programming assistant."},
            {"role": "user", "content": "Write a short Python function that adds two integers."},
        ], max_new_tokens=config.action_tokens)
        # Re-score exactly the sampled token sequence before any optimizer step.
        with torch.no_grad():
            rescored = policy.action_logprobs(generation)
            sampled = torch.tensor(generation.old_logprobs, device=rescored.device)
            mismatch = float((rescored - sampled).abs().max().item())
        # Cached generation and full BF16 forwards may differ numerically. Large
        # discrepancies invalidate importance ratios and fail the preflight.
        if mismatch > 0.05:
            raise RuntimeError(f"Generation/scoring log probabilities disagree: {mismatch:.6g}")

        # Exercise the maximum allowed context and action dimensions without
        # assuming that a naturally generated answer reaches the token ceiling.
        filler = generation.prompt_ids[-1]
        prompt_length = config.context_tokens - config.action_tokens
        stress = Generation(
            text="preflight synthetic tensor shape only",
            prompt_ids=([filler] * prompt_length),
            completion_ids=[generation.completion_ids[0]] * config.action_tokens,
            old_logprobs=[0.0] * config.action_tokens,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = -policy.action_logprobs(stress).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Maximum-shape preflight loss is not finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), 1.0,
                                                  error_if_nonfinite=True)
        optimizer.step()
        policy.version += 1
        optimizer.zero_grad(set_to_none=True)
        changed = policy.adapter_fingerprint()
        if initial == changed:
            raise RuntimeError("Preflight optimizer step did not change the adapter")
        checkpoint = run_dir / "checkpoint"
        policy.save_checkpoint(checkpoint, optimizer, metadata={"purpose": "preflight_only"})
        with torch.no_grad():
            policy.trainable_parameters()[0].view(-1)[0].add_(1)
        policy.load_checkpoint(checkpoint, optimizer)
        if policy.adapter_fingerprint() != changed:
            raise RuntimeError("Checkpoint round trip did not preserve the trained adapter")
        torch.cuda.synchronize()
        metrics = {
            "purpose": "preflight_only", "success": True,
            "model_config": asdict(config),
            "generation_seconds": generation.elapsed_seconds,
            "generated_tokens": len(generation.completion_ids),
            "old_logprob_max_abs_difference": mismatch,
            "old_logprob_difference_limit": 0.05,
            "stress_context_tokens": config.context_tokens,
            "stress_action_tokens": config.action_tokens,
            "stress_loss": float(loss.detach()), "gradient_norm": float(grad_norm),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "initial_adapter_fingerprint": initial,
            "updated_adapter_fingerprint": changed,
            "elapsed_seconds": time.monotonic() - started,
        }
        (run_dir / "preflight.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                                encoding="utf-8")
        policy.log_event("preflight_complete", **metrics)
    del loss, optimizer, policy
    gc.collect()
    torch.cuda.empty_cache()
    return metrics
