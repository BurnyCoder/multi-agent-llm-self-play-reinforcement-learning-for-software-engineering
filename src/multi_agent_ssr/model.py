"""One local, trainable policy shared by every independently prompted agent.

The Qwen3.5 BF16/text-only recipe follows the public Unsloth APIs, not copied
notebook code: https://unsloth.ai/docs/models/qwen3.5/fine-tune.md . The library
is imported only when LocalPolicy is constructed, so harness tests need no GPU.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import Generation, ModelConfig


class ContextLimitError(ValueError):
    """The task and latest complete interaction cannot fit without corruption."""


_OBSERVATION_MARKER = "\n[Observation shortened to fit context; request a narrower read. Full output is in the event log.]"


def _is_observation(message: dict[str, Any]) -> bool:
    """Recognize native tools and the harness's explicitly marked observations."""
    if message.get("role") == "tool":
        return True
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        return False
    content = message["content"]
    return ((content.startswith("Tool ") and " result: " in content)
            or content.startswith("Invalid action.")
            or (content.startswith("Subagent ") and " completed: " in content))


def _interaction_spans(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Find removable assistant/tool exchanges and reject orphan tool results.

    A tool result must stay with the assistant call that produced it. Native
    tool-call IDs are checked when present; plain textual tool protocols retain
    their enclosing assistant turn without inventing IDs.
    """
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "tool":
            raise ValueError("Tool observation has no preceding assistant turn")
        if role != "assistant":
            index += 1
            continue
        end = index + 1
        while end < len(messages) and _is_observation(messages[end]):
            end += 1
        calls = message.get("tool_calls") or []
        expected = [call.get("id") for call in calls if call.get("id")]
        received = [item.get("tool_call_id") for item in messages[index + 1:end]
                    if item.get("tool_call_id")]
        if expected and (len(expected) != len(set(expected))
                         or sorted(expected) != sorted(received)):
            raise ValueError("Assistant tool calls and tool results do not match")
        if received and not expected:
            raise ValueError("Tool result IDs have no matching assistant tool calls")
        spans.append((index, end))
        index = end
    return spans


def bound_messages(
    messages: list[dict[str, Any]],
    encode: Callable[[list[dict[str, Any]]], list[int]],
    prompt_budget: int,
) -> tuple[list[dict[str, Any]], list[int], int]:
    """Drop old exchanges, then bound observation text, preserving task instructions.

    System/task instructions and the latest assistant/tool exchange are retained.
    If necessary, shrink only observation content with an explicit marker. Tool
    envelopes and IDs stay valid; exact retained input IDs are returned for both
    logging and training. Required task text is never silently shortened.
    """
    if prompt_budget < 1:
        raise ContextLimitError("No prompt capacity remains after reserving the action")
    retained = copy.deepcopy(messages)
    removed = 0
    while True:
        spans = _interaction_spans(retained)
        ids = encode(retained)
        if len(ids) <= prompt_budget:
            return retained, ids, removed
        if len(spans) >= 2:
            start, end = spans[0]
            del retained[start:end]
            removed += 1
            continue
        observations = [(index, message["content"].removesuffix(_OBSERVATION_MARKER))
                        for index, message in enumerate(retained)
                        if _is_observation(message) and isinstance(message.get("content"), str)]
        candidates = [(index, content) for index, content in observations if len(content) > 64]
        if candidates:
            # Character prefixes preserve source indentation; re-tokenizing the
            # whole rendered prompt verifies the actual capacity after each cut.
            # Geometric shrinking avoids assuming BPE token counts are monotonic.
            index, content = max(candidates, key=lambda item: (len(item[1]), -item[0]))
            retained[index]["content"] = content[:max(64, len(content) * 3 // 4)] + _OBSERVATION_MARKER
            continue
        raise ContextLimitError(
            f"Required instructions/latest interaction use {len(ids)} tokens; "
            f"prompt capacity is {prompt_budget}"
        )


class LocalPolicy:
    """Serialize one BF16 base model, adapter, and policy version on one GPU.

    Callers performing backward/optimizer operations hold ``lock`` for the whole
    update. The same reentrant lock covers inference and checkpoint operations.
    """

    def __init__(self, config: ModelConfig, log_path: Path,
                 checkpoint: Path | None = None) -> None:
        self.config = config
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._log_lock = threading.RLock()
        self.version = 0
        self._pending_optimizer_state: dict[str, Any] | None = None
        if not config.revision:
            raise ValueError("Pin ModelConfig.revision before loading a research model")
        if config.model_id != "Qwen/Qwen3.5-2B":
            raise ValueError("This preregistered backend requires Qwen/Qwen3.5-2B")
        if not 1 <= config.context_tokens <= 2048 or not 1 <= config.action_tokens <= 256:
            raise ValueError("Pilot limits are at most 2048 context and 256 action tokens")
        if config.lora_rank != 16 or config.lora_alpha != 16:
            raise ValueError("The preregistered shared adapter uses rank=alpha=16")

        # Unsloth must be imported before Transformers/Torch patch installation.
        # Its compiler otherwise suppresses raw training logits by default:
        # https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/compiler.py
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        from unsloth import FastVisionModel  # isort: skip

        import torch

        self.torch = torch
        self._backend = FastVisionModel
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The pilot requires a CUDA GPU supporting BF16")
        self.device = torch.device("cuda:0")
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        import numpy as np
        np.random.seed(config.seed)

        self.model, self.processor = FastVisionModel.from_pretrained(
            model_name=config.model_id,
            revision=config.revision,
            # Unsloth otherwise permits mirror remapping and drops foreign SHAs:
            # https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py
            use_exact_model_name=True,
            max_seq_length=config.context_tokens,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            fast_inference=False,
            use_gradient_checkpointing="unsloth",
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        # Do not pass "all-linear": that overrides the vision=False flag in
        # https://github.com/unslothai/unsloth/blob/main/unsloth/models/vision.py .
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0,
            bias="none",
            target_modules=None,
            modules_to_save=None,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )
        self._validate_trainable_scope()
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        self.log_event("model_loaded", model=asdict(config), policy_version=self.version,
                       adapter_fingerprint=self.adapter_fingerprint(),
                       trainable_parameters=sum(p.numel() for p in self.trainable_parameters()))

    def log_event(self, event: str, **fields: Any) -> None:
        """Write complete UTF-8 events to disk and the terminal immediately."""
        record = {"timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                  "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, default=str)
        with self._log_lock:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
            try:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except UnicodeEncodeError:
                sys.stdout.buffer.write((line + "\n").encode("utf-8"))
                sys.stdout.buffer.flush()

    def _validate_trainable_scope(self) -> None:
        """Reject accidental full-weight, embedding, head, or vision training."""
        names = [name for name, parameter in self.model.named_parameters()
                 if parameter.requires_grad]
        forbidden = ("visual", "vision", "embed_tokens", "lm_head")
        if not names or any("lora_" not in name or any(part in name for part in forbidden)
                            for name in names):
            raise RuntimeError("Only language-layer LoRA parameters may be trainable")
        # A separate adapter per role would violate the parameter-sharing design.
        if set(getattr(self.model, "peft_config", {})) != {"default"}:
            raise RuntimeError("Exactly one shared 'default' adapter is required")

    def trainable_parameters(self) -> list[Any]:
        """Return the single optimizer's trainable parameter objects."""
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def adapter_fingerprint(self) -> str:
        """Hash adapter names, shapes and exact bytes, independent of device."""
        with self.lock:
            digest = hashlib.sha256()
            for name, parameter in sorted(self.model.named_parameters()):
                if parameter.requires_grad:
                    value = parameter.detach().cpu().contiguous()
                    digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode())
                    digest.update(value.view(self.torch.uint8).numpy().tobytes())
            return digest.hexdigest()

    def _encode_messages(self, messages: list[dict[str, Any]],
                         tools: list[dict[str, Any]] | None) -> list[int]:
        """Render the model's own template without lossy decode/re-encode cycles."""
        owner = self.processor if hasattr(self.processor, "apply_chat_template") else self.tokenizer
        rendered = owner.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer.encode(rendered, add_special_tokens=False)

    def generate(self, messages: list[dict[str, Any]], *, max_new_tokens: int,
                 tools: list[dict[str, Any]] | None = None) -> Generation:
        """Sample once, preserving exact action IDs and sampling log probabilities.

        The recorder retains only one vocabulary-sized vector between steps;
        unlike output_scores=True, memory does not grow by a vocabulary per token.
        Transformers custom logits processors run before multinomial sampling:
        https://huggingface.co/docs/transformers/main_classes/text_generation .
        """
        from transformers import (
            GenerationConfig,
            LogitsProcessor,
            LogitsProcessorList,
            TextStreamer,
        )

        if not 1 <= max_new_tokens <= self.config.action_tokens:
            raise ValueError(f"max_new_tokens must be between 1 and {self.config.action_tokens}")
        with self.lock:
            started = time.monotonic()
            self.log_event("prompt_requested", messages=messages, tools=tools,
                           policy_version=self.version)
            retained, prompt_ids, removed = bound_messages(
                messages, lambda current: self._encode_messages(current, tools),
                self.config.context_tokens - max_new_tokens,
            )
            self.log_event("prompt_effective", messages=retained, prompt_ids=prompt_ids,
                           removed_exchanges=removed, policy_version=self.version)
            policy = self

            class Recorder(LogitsProcessor):
                """Capture the probability of the previous sampled token next turn."""

                def __init__(self) -> None:
                    self.previous = None
                    self.values: list[float] = []

                def __call__(self, input_ids: Any, scores: Any) -> Any:
                    if self.previous is not None:
                        self.values.append(float(self.previous[0, input_ids[0, -1]].item()))
                    self.previous = scores.detach().float().log_softmax(dim=-1)
                    return scores

                def finish(self, ids: list[int]) -> list[float]:
                    if ids and self.previous is not None:
                        self.values.append(float(self.previous[0, ids[-1]].item()))
                    self.previous = None
                    if len(ids) != len(self.values):
                        raise RuntimeError("Sampling log-probability alignment failed")
                    return self.values

            class Streamer(TextStreamer):
                """Preserve every decoded chunk in the timestamped event stream."""

                def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                    if text or stream_end:
                        policy.log_event("output_chunk", text=text, stream_end=stream_end,
                                         policy_version=policy.version)

            recorder = Recorder()
            input_ids = self.torch.tensor([prompt_ids], device=self.device, dtype=self.torch.long)
            defaults = self.model.generation_config
            eos = defaults.eos_token_id
            pad = self.tokenizer.pad_token_id
            if pad is None:
                pad = eos[0] if isinstance(eos, list) else eos
            # Construct a fresh config: inherited top-k or repetition penalties
            # would make the scored policy differ from the sampling distribution.
            generation_config = GenerationConfig(
                do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                eos_token_id=eos, pad_token_id=pad,
                bos_token_id=defaults.bos_token_id, use_cache=True,
                output_scores=False, return_dict_in_generate=False,
            )
            self._backend.for_inference(self.model)
            try:
                with self.torch.inference_mode():
                    output = self.model.generate(
                        input_ids=input_ids, attention_mask=self.torch.ones_like(input_ids),
                        generation_config=generation_config,
                        logits_processor=LogitsProcessorList([recorder]),
                        streamer=Streamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True),
                    )
                completion_ids = output[0, len(prompt_ids):].tolist()
                old_logprobs = recorder.finish(completion_ids)
            finally:
                self._backend.for_training(self.model)
            if not completion_ids or not all(math.isfinite(value) for value in old_logprobs):
                raise RuntimeError("Model produced no valid finite-probability action")
            result = Generation(
                text=self.tokenizer.decode(completion_ids, skip_special_tokens=True),
                prompt_ids=prompt_ids, completion_ids=completion_ids,
                old_logprobs=old_logprobs, policy_version=self.version,
                elapsed_seconds=time.monotonic() - started,
            )
            self.log_event("generation_complete", **asdict(result))
            return result

    def action_logprobs(self, generation: Generation) -> Any:
        """Score only sampled continuation targets, retaining a gradient graph.

        logits_to_keep avoids allocating a context-length by vocabulary tensor.
        The extra preceding position predicts the first completion token; the
        final next-token logit is discarded. The 32-token chunks avoid a second
        full float32 vocabulary allocation. The model API is documented in:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
        """
        torch = self.torch
        if not generation.prompt_ids or not generation.completion_ids:
            raise ValueError("A scored action requires both prompt and completion IDs")
        if len(generation.prompt_ids) + len(generation.completion_ids) > self.config.context_tokens:
            raise ContextLimitError("Stored action exceeds the registered context size")
        with self.lock:
            self._backend.for_training(self.model)
            ids = torch.tensor([generation.prompt_ids + generation.completion_ids],
                               dtype=torch.long, device=self.device)
            count = len(generation.completion_ids)
            output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                use_cache=False, logits_to_keep=count + 1)
            logits = output.logits
            if not isinstance(logits, torch.Tensor) or logits.shape[1] != count + 1:
                raise RuntimeError("Backend did not honor action-only logits_to_keep")
            targets = ids[0, -count:]
            chunks = []
            for start in range(0, count, 32):
                end = min(start + 32, count)
                rows = logits[0, start:end].float()
                selected = rows.gather(1, targets[start:end, None]).squeeze(1)
                chunks.append(selected - torch.logsumexp(rows, dim=-1))
            return torch.cat(chunks)

    def save_checkpoint(self, path: Path, optimizer: Any | None = None,
                        *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Save one adapter plus optimizer and exact RNG states, never credentials."""
        with self.lock:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(path, safe_serialization=True)
            self.processor.save_pretrained(path)
            record = {"format_version": 1, "model_config": asdict(self.config),
                      "policy_version": self.version,
                      "adapter_fingerprint": self.adapter_fingerprint(),
                      "metadata": metadata or {}}
            (path / "checkpoint.json").write_text(json.dumps(record, indent=2) + "\n",
                                                   encoding="utf-8")
            import numpy as np
            numpy_state = np.random.get_state()
            state = {
                "optimizer": optimizer.state_dict() if optimizer is not None else self._pending_optimizer_state,
                "python_rng": random.getstate(),
                "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                "torch_rng": self.torch.get_rng_state(),
                "cuda_rng": self.torch.cuda.get_rng_state_all(),
            }
            self.torch.save(state, path / "training_state.pt")
            self.log_event("checkpoint_saved", path=str(path), **record)
            return record

    def load_checkpoint(self, path: Path, optimizer: Any | None = None,
                        *, restore_rng: bool = True) -> dict[str, Any]:
        """Restore into the existing adapter instead of creating another policy."""
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights

        with self.lock:
            path = Path(path)
            record = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
            if record.get("format_version") != 1 or record["model_config"] != asdict(self.config):
                raise ValueError("Checkpoint configuration does not match this policy")
            weights = load_peft_weights(str(path), device="cpu")
            set_peft_model_state_dict(self.model, weights, adapter_name="default")
            self._validate_trainable_scope()
            self.version = int(record["policy_version"])
            if self.adapter_fingerprint() != record["adapter_fingerprint"]:
                raise RuntimeError("Restored adapter differs from the saved fingerprint")
            state = self.torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
            if optimizer is not None and state["optimizer"] is not None:
                optimizer.load_state_dict(state["optimizer"])
            else:
                self._pending_optimizer_state = state["optimizer"]
            if restore_rng:
                random.setstate(state["python_rng"])
                import numpy as np
                rng = state["numpy_rng"]
                np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))
                self.torch.set_rng_state(state["torch_rng"])
                self.torch.cuda.set_rng_state_all(state["cuda_rng"])
            self.log_event("checkpoint_loaded", path=str(path), policy_version=self.version,
                           adapter_fingerprint=record["adapter_fingerprint"])
            return record
