import subprocess
from types import SimpleNamespace

from llm_wiki.v2 import llm_adapter


def test_command_adapter_retries_a_subprocess_timeout_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_adapter, "MAX_RETRIES", 3)
    monkeypatch.setattr(llm_adapter.time, "sleep", lambda seconds: None)
    attempts = {"count": 0}

    def flaky_run(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise subprocess.TimeoutExpired(cmd="bridge", timeout=180)
        return SimpleNamespace(returncode=0, stdout='{"concepts": []}', stderr="")

    monkeypatch.setattr(llm_adapter.subprocess, "run", flaky_run)
    adapter = llm_adapter.CommandUserLLMAdapter("bridge")
    result = adapter._call("extract_concepts", {"chunk": "x"})
    assert result == {"concepts": []}
    assert attempts["count"] == 3


def test_command_adapter_retries_a_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_adapter, "MAX_RETRIES", 3)
    monkeypatch.setattr(llm_adapter.time, "sleep", lambda seconds: None)
    attempts = {"count": 0}

    def flaky_run(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            return SimpleNamespace(returncode=1, stdout="", stderr="rate limited")
        return SimpleNamespace(returncode=0, stdout='{"concepts": []}', stderr="")

    monkeypatch.setattr(llm_adapter.subprocess, "run", flaky_run)
    adapter = llm_adapter.CommandUserLLMAdapter("bridge")
    result = adapter._call("extract_concepts", {"chunk": "x"})
    assert result == {"concepts": []}
    assert attempts["count"] == 3


def test_command_adapter_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm_adapter, "MAX_RETRIES", 2)
    monkeypatch.setattr(llm_adapter.time, "sleep", lambda seconds: None)
    attempts = {"count": 0}

    def always_fails(*args, **kwargs):
        attempts["count"] += 1
        return SimpleNamespace(returncode=1, stdout="", stderr="down")

    monkeypatch.setattr(llm_adapter.subprocess, "run", always_fails)
    adapter = llm_adapter.CommandUserLLMAdapter("bridge")
    try:
        adapter._call("extract_concepts", {"chunk": "x"})
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert attempts["count"] == 2


def test_agent_adapter_retries_a_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_adapter, "MAX_RETRIES", 3)
    monkeypatch.setattr(llm_adapter.time, "sleep", lambda seconds: None)
    attempts = {"count": 0}

    def mock_which(agent):
        return f"/usr/bin/{agent}"

    def flaky_run(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            return SimpleNamespace(returncode=1, stdout="", stderr="rate limited")
        return SimpleNamespace(returncode=0, stdout='{"result": "{\\"type\\": \\"object\\"}"}', stderr="")

    monkeypatch.setattr(llm_adapter.shutil, "which", mock_which)
    monkeypatch.setattr(llm_adapter.subprocess, "run", flaky_run)
    adapter = llm_adapter.AgentCLIUserLLMAdapter("claude")
    result = adapter._call("classify_relation", {"json_schema": {"type": "object"}})
    assert result == {"type": "object"}
    assert attempts["count"] == 3
