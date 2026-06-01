from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re

import pytest

from src.tools.python_code_execution.tool import run_python_code_tool


def invoke_tool(file_content: str, argv=None, timeout_s: int = 5) -> str:
    """Helper to invoke the tool whether it's a tool wrapper or plain function."""
    if hasattr(run_python_code_tool, "invoke"):
        return run_python_code_tool.invoke({"file_content": file_content, "argv": argv or [], "timeout_s": timeout_s})
    # fallback: call directly
    return run_python_code_tool(file_content=file_content, argv=argv or [], timeout_s=timeout_s)


def test_execution_prints_stdout():
    output = invoke_tool("print('hello_world')")
    assert "hello_world" in output


def test_execution_returns_error_on_exception():
    # Code that raises an exception should result in an error string containing 'Execution failed'
    output = invoke_tool("raise RuntimeError('boom')")
    assert "Execution failed with exit code" in output
    assert "RuntimeError" in output


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
