"""
    This is an agent tool that runs python code. It takes in the whole content of the python file and the parameters to run the code with. We will temporarily create a file with the content of the python code and then run the file with the parameters as input. The output of the code will be returned as a string. The file will be deleted after running the code. 
    
    It is important that the file has a __main__ function that takes in the parameters as input and runs the code. The tool will run the code and return the output of the code. 
"""
from __future__ import annotations

import subprocess
import tempfile
from typing import List

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class RunPythonCodeInput(BaseModel):
    file_content: str = Field(..., description="Full contents of the Python file to execute.")
    argv: List[str] = Field(default_factory=list, description="Command-line args passed to the script.")
    timeout_s: int = Field(10, ge=1, le=120, description="Timeout in seconds.")


@tool("run_python_code_tool", args_schema=RunPythonCodeInput)
def run_python_code_tool(file_content: str, argv: List[str] = [], timeout_s: int = 10) -> str:
    """Write `file_content` to a temp .py file, run it, and return combined stdout/stderr."""
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/_tmp_run.py"
        with open(path, "w", encoding="utf-8") as f:
            f.write(file_content)

        proc = subprocess.run(
            ["python", path, *argv],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return (proc.stdout or "") + (proc.stderr or "")

if __name__ == "__main__":
    # Example usage
    filecontents = (
    "def main(params):\n" +
    "    name = params.get(\"name\", \"World\")\n" +
    "    return f\"Hello, {name}!\"\n" +
    "if __name__ == \"__main__\":\n" +
    "    import sys\n" +
    "    import json\n" +
    "    params = json.loads(sys.stdin.read())\n" +
    "    print(main(params))"
    )
    output = run_python_code_tool.invoke({"filecontents": filecontents, "parameters": {"name": "Areeb"}})
    print(output)