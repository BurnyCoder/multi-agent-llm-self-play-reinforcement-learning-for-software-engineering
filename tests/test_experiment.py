"""Stopping and configuration rules independent of expensive model runs."""
from pathlib import Path


def test_budget_survives_restart(tmp_path: Path):
    from multi_agent_ssr.experiment import Budget
    first = Budget(tmp_path, 28800, now=100)
    second = Budget(tmp_path, 28800, now=200)
    assert first.deadline == second.deadline == 28900
    assert second.remaining(now=200) == 28700


def test_config_inheritance_does_not_mutate_base(tmp_path: Path):
    from multi_agent_ssr.experiment import load_config
    (tmp_path / "base.toml").write_text('[model]\nseed=0\n[experiment]\ndelegation=true\n')
    (tmp_path / "child.toml").write_text('extends="base.toml"\n[experiment]\ndelegation=false\n')
    assert not load_config(tmp_path / "child.toml")["experiment"]["delegation"]
    assert load_config(tmp_path / "base.toml")["experiment"]["delegation"]


def test_budget_configuration_cannot_reset_existing_run(tmp_path: Path):
    import pytest

    from multi_agent_ssr.experiment import Budget
    Budget(tmp_path, 10, now=100)
    with pytest.raises(ValueError, match="budget"):
        Budget(tmp_path, 100, now=100)
