"""Concurrent primary/child sessions with a single shared-policy backend.

Delegation follows OpenCode's fresh child session and resumable-session design:
https://github.com/anomalyco/opencode/blob/e761e265d2d5e6ff4a1c0756ab8d8eeee0db501e/packages/core/src/tool/plugin/subagent.ts
The GPU backend serializes inference, but parents and children execute tools and
advance independently. Every actual generation is recorded exactly once.
"""

from __future__ import annotations

import json
import queue
import re
import shlex
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import ActionTrace, Episode, Limits, Policy
from .io import EventLog, utc_now
from .sandbox import SandboxInfrastructureError


def parse_action(text: str) -> dict[str, Any] | None:
    """Parse one native Qwen call or a JSON call, without executing generated text.

    Qwen uses XML-like delimiters with raw, potentially unescaped code values:
    https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/chat_template.jinja
    This is not XML; only the explicit delimiter grammar is interpreted. String
    arguments preserve code whitespace; schema integers/arrays use JSON decoding.
    """
    candidate = text.strip()
    if "<tool_call>" in candidate or "</tool_call>" in candidate:
        if candidate.count("<tool_call>") != 1 or candidate.count("</tool_call>") != 1:
            return None
        wrapped = re.fullmatch(r".*?<tool_call>\s*(.*?)\s*</tool_call>", candidate, re.S)
        if wrapped is None:
            return None
        candidate = wrapped.group(1)
    elif candidate.startswith("```"):
        fenced = re.fullmatch(r"```json\s*(.*?)\s*```", candidate, re.S)
        if fenced is None:
            return None
        candidate = fenced.group(1)
    schemas = {tool["function"]["name"]: tool["function"]["parameters"]
               for tool in tool_definitions("primary", True)}
    if candidate.startswith("<function="):
        function = re.fullmatch(r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>", candidate, re.S)
        if function is None:
            return None
        name, remaining = function.groups()
        properties = schemas.get(name, {}).get("properties", {})
        arguments = {}
        while remaining.strip():
            parameter = re.match(r"\s*<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>", remaining, re.S)
            if parameter is None:
                return None
            key, value = parameter.groups()
            if key in arguments or re.search(r"</?(?:parameter|function|tool_call)(?:=|>)", value):
                return None
            # The native template contributes exactly one newline at each end.
            value = re.sub(r"\A\r?\n|\r?\n\Z", "", value)
            if properties.get(key, {}).get("type") in {"integer", "array"}:
                try:
                    value = json.loads(value)
                except (ValueError, TypeError):
                    return None
            arguments[key] = value
            remaining = remaining[parameter.end():]
        result = {"name": name, "arguments": arguments}
    else:
        try:
            result = json.loads(candidate)
        except (ValueError, TypeError):
            return None
    if not isinstance(result, dict) or not isinstance(result.get("name"), str):
        return None
    arguments = result.get("arguments", {})
    if not isinstance(arguments, dict) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", result["name"]):
        return None
    schema = schemas.get(result["name"], {})
    if any(key not in arguments for key in schema.get("required", [])):
        return None
    for key, value in arguments.items():
        kind = schema.get("properties", {}).get(key, {}).get("type")
        if ((kind == "string" and not isinstance(value, str))
                or (kind == "integer" and type(value) is not int)
                or (kind == "array" and (not isinstance(value, list)
                                        or any(not isinstance(item, str) for item in value)))):
            return None
    return {"name": result["name"], "arguments": arguments}


def tool_definitions(role: str, delegation: bool) -> list[dict[str, Any]]:
    """Tool descriptions are compact enough for the preregistered small context."""
    definitions = [
        ("read", "Read a repository file (numbered lines).", {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
        ("search", "Search repository text using a literal pattern.", {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
        ("finish", "Finish your work; injector must list selected test paths.", {"summary": {"type": "string"}, "test_targets": {"type": "array", "items": {"type": "string"}}}, ["summary"]),
    ]
    if role != "explore":
        definitions += [
            ("edit", "Replace exact text once; old='' creates a new file only.", {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, ["path", "old", "new"]),
            ("bash", "Run a command in the isolated repository sandbox.", {"command": {"type": "string"}}, ["command"]),
        ]
    if role == "primary" and delegation:
        definitions.append(("subagent", "Launch asynchronous general (edit/test) or explore (read-only) work. Continue independent work while it runs. Results arrive automatically. Supply session_id to follow up; give complete context.", {"agent": {"type": "string", "enum": ["general", "explore"]}, "description": {"type": "string"}, "prompt": {"type": "string"}, "session_id": {"type": "string"}}, ["agent", "description", "prompt"]))
    return [{"type": "function", "function": {"name": name, "description": description,
             "parameters": {"type": "object", "properties": properties, "required": required}}}
            for name, description, properties, required in definitions]


@dataclass
class Session:
    """Independent context plus a mailbox; only its own worker changes messages."""

    session_id: str
    role: str
    parent_id: str | None
    messages: list[dict[str, Any]]
    mailbox: queue.Queue = field(default_factory=queue.Queue)
    running: bool = False
    result: dict[str, Any] = field(default_factory=dict)


class AgentHarness:
    """Own one team's asynchronous graph and enforce shared resource budgets."""

    def __init__(self, policy: Policy, sandbox: Any, team: str, task_id: str,
                 log_dir: Path, limits: Limits, delegation: bool):
        self.policy, self.sandbox, self.limits = policy, sandbox, limits
        self.delegation = delegation
        self.episode = Episode(uuid.uuid4().hex, task_id, team)
        self.log = EventLog(log_dir / "events.jsonl")
        self.lock, self.edit_lock = threading.RLock(), threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.sessions: dict[str, Session] = {}
        self.reserved_tokens = self.used_tokens = self.used_tools = 0
        self.deadline = time.monotonic() + limits.episode_seconds
        self.executor = ThreadPoolExecutor(max_workers=max(1, limits.child_sessions))

    def _new_session(self, role: str, prompt: str, parent: str | None) -> Session:
        """A child inherits role constraints, never the parent's full transcript."""
        session_id = uuid.uuid4().hex[:12]
        system = (f"You are a software engineering agent. SESSION ROLE: {role}. "
                  f"TEAM: {self.episode.team}. Work in /repo. Use tools to inspect code before editing. "
                  "Emit one tool call at a time. Read small file ranges. Do not modify test infrastructure. "
                  "Use finish when done. ")
        if role == "primary" and self.delegation:
            system += "Delegate independent investigation or implementation to subagents; continue your own non-overlapping work while they run. "
        if role == "explore":
            system += "You may only search and read, never change files. "
        session = Session(session_id, role, parent, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        self.sessions[session_id] = session
        self.log.emit("session_started", episode_id=self.episode.episode_id,
                      session_id=session_id, parent_session_id=parent, role=role, team=self.episode.team)
        return session

    def _reserve(self) -> int:
        """Reserve before inference so simultaneous requests cannot exceed budget."""
        with self.lock:
            remaining = self.limits.generated_tokens - self.used_tokens - self.reserved_tokens
            if time.monotonic() >= self.deadline or remaining <= 0 or self.used_tools >= self.limits.tool_calls:
                return 0
            count = min(256, remaining)
            self.reserved_tokens += count
            return count

    def _child(self, parent: Session, args: dict[str, Any]) -> dict[str, Any]:
        """Return immediately after launch; an existing session accepts follow-ups."""
        if parent.parent_id or not self.delegation:
            raise ValueError("Nested delegation is disabled")
        role = args.get("agent")
        if role not in {"general", "explore"}:
            raise ValueError("Unknown subagent type")
        with self.condition:
            session_id = args.get("session_id")
            if session_id:
                child = self.sessions.get(session_id)
                if child is None or child.parent_id != parent.session_id or child.role != role:
                    raise ValueError("Session is not this parent's child of the requested type")
                child.mailbox.put(str(args["prompt"]))
            else:
                if sum(s.parent_id is not None for s in self.sessions.values()) >= self.limits.child_sessions:
                    raise ValueError("Child session budget exhausted")
                child = self._new_session(role, str(args["prompt"]), parent.session_id)
            if not child.running:
                child.running = True
                self.executor.submit(self._run_session, child)
            return {"session_id": child.session_id, "status": "running", "message": "Continue independent work; completion arrives automatically."}

    def _tool(self, session: Session, name: str, args: dict[str, Any]) -> Any:
        """Dispatch only exposed tools; explore cannot smuggle writes via shell."""
        allowed = {tool["function"]["name"] for tool in tool_definitions(session.role, self.delegation)}
        if name not in allowed:
            raise ValueError(f"Tool {name} is unavailable for {session.role}")
        if name == "subagent":
            return self._child(session, args)
        if name == "read":
            lines = self.sandbox.read(str(args["path"])).splitlines()
            start = max(1, int(args.get("start_line", 1)))
            end = min(len(lines), start + 60, int(args.get("end_line", start + 30)))
            return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines[start - 1:end], start - 1))
        if name == "search":
            pattern, path = str(args["pattern"]), str(args.get("path", "."))
            return self.sandbox.exec(f"grep -RFn --exclude-dir=.git -- {shlex.quote(pattern)} {shlex.quote(path)} | head -60", timeout=self.limits.command_seconds)
        if name == "bash":
            return self.sandbox.exec(str(args["command"]), timeout=self.limits.command_seconds)
        if name == "edit":
            with self.edit_lock:
                path, old, new = str(args["path"]), str(args["old"]), str(args["new"])
                try:
                    previous = self.sandbox.read(path)
                except SandboxInfrastructureError:
                    raise
                except (FileNotFoundError, ValueError, RuntimeError):
                    previous = None
                if old == "":
                    if previous is not None:
                        raise ValueError("Refusing to overwrite existing file with empty old text")
                    updated = new
                elif previous is None or previous.count(old) != 1:
                    raise ValueError("Old text must occur exactly once; re-read after concurrent edits")
                else:
                    updated = previous.replace(old, new, 1)
                self.sandbox.write(path, updated)
            return {"edited": path}
        if name == "finish":
            return args
        raise ValueError("Unknown tool")

    def _run_session(self, session: Session) -> None:
        """Consume asynchronous notifications between actual model calls."""
        try:
            while True:
                while not session.mailbox.empty():
                    session.messages.append({"role": "user", "content": session.mailbox.get_nowait()})
                reserved = self._reserve()
                if not reserved:
                    session.result = {"summary": "Episode resource budget exhausted", "budget_exhausted": True}
                    break
                try:
                    generation = self.policy.generate(session.messages, max_new_tokens=reserved,
                                                      tools=tool_definitions(session.role, self.delegation))
                except Exception:
                    with self.lock:
                        self.reserved_tokens -= reserved
                    raise
                action_trace = ActionTrace(uuid.uuid4().hex, session.session_id, session.parent_id,
                                           session.role, generation, utc_now())
                with self.lock:
                    self.reserved_tokens -= reserved
                    self.used_tokens += len(generation.completion_ids)
                    self.episode.actions.append(action_trace)
                self.log.emit("generation", episode_id=self.episode.episode_id,
                              **{**action_trace.__dict__, "generation": generation.__dict__})
                action = parse_action(generation.text)
                session.messages.append({"role": "assistant", "content": generation.text})
                if action is None:
                    session.messages.append({"role": "user", "content": "Invalid action. Emit one tool call, for example <tool_call><function=read><parameter=path>README.md</parameter></function></tool_call>."})
                    continue
                with self.lock:
                    if self.used_tools >= self.limits.tool_calls:
                        session.result = {"summary": "Team tool budget exhausted", "budget_exhausted": True}
                        break
                    self.used_tools += 1
                name, args = action["name"], action.get("arguments", {})
                self.log.emit("tool_started", session_id=session.session_id, name=name, arguments=args)
                try:
                    result = self._tool(session, name, args)
                except SandboxInfrastructureError:
                    raise
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                self.log.emit("tool_finished", session_id=session.session_id, name=name, result=result)
                if name == "finish" and "error" not in result:
                    session.result = result
                    if session.parent_id is None:
                        with self.condition:
                            if any(s.running for s in self.sessions.values() if s.parent_id):
                                self.condition.wait_for(lambda: not any(s.running for s in self.sessions.values() if s.parent_id),
                                                        timeout=max(0, self.deadline - time.monotonic()))
                                if not session.mailbox.empty() and self.used_tokens < self.limits.generated_tokens:
                                    continue
                    break
                # Tool observations are context only. Raw outputs remain unabridged in logs.
                observation = json.dumps(result, ensure_ascii=False)
                if len(observation) > 3000:
                    observation = observation[:3000] + "\n[Observation shortened; request a narrower read. Full output is in the event log.]"
                session.messages.append({"role": "user", "content": f"Tool {name} result: {observation}"})
        except Exception as exc:
            session.result = {"error": f"{type(exc).__name__}: {exc}"}
            with self.lock:
                # A second worker reaching its context limit must not hide an
                # earlier device/runtime failure from the trusted scorer.
                if self.episode.error is None or self.episode.error.startswith("ContextLimitError:"):
                    self.episode.error = session.result["error"]
        finally:
            with self.condition:
                session.running = False
                if session.parent_id:
                    self.sessions[session.parent_id].mailbox.put(f"Subagent {session.session_id} completed: {json.dumps(session.result)}")
                self.log.emit("session_finished", session_id=session.session_id, parent_session_id=session.parent_id,
                              result=session.result)
                self.condition.notify_all()

    def run(self, prompt: str) -> Episode:
        """Do not update weights or score a mutable sandbox until all workers end."""
        started = time.monotonic()
        root = self._new_session("primary", prompt, None)
        root.running = True
        self._run_session(root)
        self.executor.shutdown(wait=True)
        self.episode.result = root.result
        self.episode.completed = (self.episode.error is None or
                                  self.episode.error.startswith("ContextLimitError:"))
        self.episode.elapsed_seconds = time.monotonic() - started
        return self.episode


def run_team(policy: Policy, sandbox: Any, team: str, task_id: str, prompt: str,
             log_dir: Path, *, limits: Limits | None = None, delegation: bool = True) -> Episode:
    """Public rollout entry point; orchestration owns the sandbox lifecycle."""
    return AgentHarness(policy, sandbox, team, task_id, log_dir, limits or Limits(), delegation).run(prompt)
