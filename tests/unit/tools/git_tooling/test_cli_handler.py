import types
import subprocess
import pytest
from pathlib import Path
import src.settings as settings_module
import src.tools.git_tooling.command_line.cli_handler as cli_handler

@pytest.fixture
def temp_workplace(tmp_path):
    wp = tmp_path / "workplace"
    wp.mkdir()
    # set global settings object values
    monkey_target = settings_module.settings
    return wp, monkey_target

@pytest.fixture
def tmp_repo(tmp_path):
    wp = tmp_path / "workplace"
    wp.mkdir()
    repo = wp / "myrepo"
    repo.mkdir()
    (repo / ".git").mkdir()  # simulate repo
    print("Temp repo path:", repo)
    return repo, wp

from types import SimpleNamespace

def test_run_git_command_success(tmp_repo, monkeypatch):
    repo, wp = tmp_repo
    print("Testing with repo path:", repo)
    # set settings
    import src.settings as settings_module
    monkeypatch.setattr(settings_module.settings, "WORKPLACE_FOLDER", str(wp))

    # patch subprocess.run to simulate success
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="git OK\n", stderr="")
    monkeypatch.setattr("src.tools.git_tooling.command_line.cli_handler.subprocess.run", fake_run)

    cli = cli_handler.GitCLI(str(repo))
    out = cli.status()
    assert out.get("success") is True
    assert "git OK" in out.get("stdout", "")

def test_run_git_command_timeout(tmp_repo, monkeypatch):
    repo, wp = tmp_repo
    import src.settings as settings_module
    monkeypatch.setattr(settings_module.settings, "WORKPLACE_FOLDER", str(wp))

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)
    monkeypatch.setattr("src.tools.git_tooling.command_line.cli_handler.subprocess.run", raise_timeout)

    cli = cli_handler.GitCLI(str(repo))
    with pytest.raises(cli_handler.GitCLIError):
        cli.status()
