"""Behavioral tests for asynchronous delegation and shared episode accounting."""

import json
import threading
from pathlib import Path

import pytest

from multi_agent_ssr.contracts import Generation, Limits


class FakeSandbox:
    def __init__(self):
        self.parent_work = threading.Event()

    def read(self, path):
        self.parent_work.set()
        return "def add(a, b): return a + b"

    def exec(self, command, timeout=90):
        return {"stdout": "", "stderr": "", "returncode": 0}

    def write(self, path, content):
        pass


class ScriptedPolicy:
    version = 0

    def __init__(self, sandbox, native=False):
        self.sandbox = sandbox
        self.root_calls = 0
        self.native = native

    def generate(self, messages, *, max_new_tokens, tools=None):
        if "SESSION ROLE: primary" in messages[0]["content"]:
            self.root_calls += 1
            if self.root_calls == 1:
                action = {"name": "subagent", "arguments": {"agent": "explore", "description": "Inspect code", "prompt": "Find addition."}}
            elif self.root_calls == 2:
                action = {"name": "read", "arguments": {"path": "src.py"}}
            else:
                action = {"name": "finish", "arguments": {"summary": "Done", "test_targets": ["tests"]}}
        else:
            assert self.sandbox.parent_work.wait(3), "parent was blocked by child"
            action = {"name": "finish", "arguments": {"summary": "Found addition"}}
        if self.native:
            parameters = "".join(f'<parameter={key}>\n{json.dumps(value) if isinstance(value, list) else value}\n</parameter>\n'
                                 for key, value in action["arguments"].items())
            text = f'<tool_call>\n<function={action["name"]}>\n{parameters}</function>\n</tool_call>'
        else:
            text = '<tool_call>' + json.dumps(action) + '</tool_call>'
        return Generation(text, [1], [2, 3], [-0.1, -0.2])


@pytest.mark.parametrize("native", [False, True])
def test_parent_continues_while_child_runs(tmp_path: Path, native):
    from multi_agent_ssr.agents import run_team

    sandbox = FakeSandbox()
    policy = ScriptedPolicy(sandbox, native=native)
    episode = run_team(policy, sandbox, "repair", "task", "Repair it", tmp_path,
                       limits=Limits(episode_seconds=10))
    assert episode.completed
    assert sandbox.parent_work.is_set()
    assert any(a.parent_session_id for a in episode.actions)
    assert episode.token_count == 2 * len(episode.actions)
    assert {a.generation.policy_version for a in episode.actions} == {0}


def test_parse_actions_does_not_execute_text():
    from multi_agent_ssr.agents import parse_action

    assert parse_action('text only') is None
    assert parse_action('<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>')["name"] == "read"


def test_explore_has_no_mutating_tools():
    from multi_agent_ssr.agents import tool_definitions

    names = {tool["function"]["name"] for tool in tool_definitions("explore", True)}
    assert names == {"read", "search", "finish"}


def test_parse_native_qwen_call_preserves_edit_whitespace_and_code():
    """Qwen's parameter delimiters add one newline around raw string values."""
    from multi_agent_ssr.agents import parse_action

    text = ('I will fix the comparison.\n<tool_call>\n<function=edit>\n'
            '<parameter=path>\nsrc.py\n</parameter>\n'
            '<parameter=old>\n    if value < 1:\n        return "true"\n</parameter>\n'
            '<parameter=new>\n    if value <= 1:\n        return "false"\n</parameter>\n'
            '</function>\n</tool_call>')
    assert parse_action(text) == {"name": "edit", "arguments": {
        "path": "src.py", "old": '    if value < 1:\n        return "true"',
        "new": '    if value <= 1:\n        return "false"',
    }}


def test_parse_native_qwen_converts_schema_numbers_and_arrays_only():
    from multi_agent_ssr.agents import parse_action

    read = ('<tool_call><function=read><parameter=path>true</parameter>'
            '<parameter=start_line>12</parameter><parameter=end_line>24</parameter>'
            '</function></tool_call>')
    assert parse_action(read)["arguments"] == {"path": "true", "start_line": 12, "end_line": 24}
    finish = ('<tool_call><function=finish><parameter=summary>Done</parameter>'
              '<parameter=test_targets>["tests/test_a.py", "tests/test_b.py"]</parameter>'
              '</function></tool_call>')
    assert parse_action(finish)["arguments"]["test_targets"] == ["tests/test_a.py", "tests/test_b.py"]


@pytest.mark.parametrize("text", [
    '<tool_call><function=read><parameter=path>a.py</parameter></function>',
    '<tool_call><function=read><parameter=path>a.py</parameter></function></tool_call>suffix',
    '<tool_call><function=read><parameter=path>a.py</parameter></function></tool_call>' * 2,
    '<tool_call><function=read><parameter=path>a.py</parameter><parameter=path>b.py</parameter></function></tool_call>',
    '<tool_call><function=read>unparsed text<parameter=path>a.py</parameter></function></tool_call>',
    '<tool_call><function=read><parameter=path>a.py</wrong></function></tool_call>',
    '<tool_call><function=read><parameter=path>a.py</parameter><parameter=start_line>true</parameter></function></tool_call>',
    '<tool_call><function=finish><parameter=summary>Done</parameter><parameter=test_targets>tests/test_a.py</parameter></function></tool_call>',
    '<tool_call><function=finish><parameter=summary>Done</parameter><parameter=test_targets>[1]</parameter></function></tool_call>',
])
def test_parse_native_qwen_rejects_malformed_or_ambiguous_actions(text):
    from multi_agent_ssr.agents import parse_action

    assert parse_action(text) is None
