"""
This is an agent tool that runs python code from temporary files.
It accepts the whole file content, optional CLI args, and a timeout.
"""
from langchain_core.tools import tool
import os
import subprocess
import sys
import tempfile
from typing import List


@tool
def run_python_code_tool(
    file_content: str,
    argv: List[str] | None = None,
    timeout_s: int = 10,
) -> str:
    """
    Runs the given python code and returns stdout as a string.

    Args:
        file_content (str): The whole content of the python file to be run.
        argv (List[str] | None): Optional CLI arguments for the script.
        timeout_s (int): Maximum execution time in seconds.

    Returns:
        str: The output of the code as a string.
    """
    if argv is None:
        argv = []

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(file_content.encode())
            temp_file_path = temp_file.name

        result = subprocess.run(
            [sys.executable, temp_file_path, *argv],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )

        if result.returncode != 0:
            return (
                f"Execution failed with exit code {result.returncode}.\n"
                f"STDERR:\n{result.stderr.strip()}\n"
                f"STDOUT:\n{result.stdout.strip()}"
            )

        return result.stdout
    finally:
        if temp_file_path is not None and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    # Example usage
    file_content = (
    "import sys\n" +
    "def main(args):\n" +
    "    name = args[0] if args else \"World\"\n" +
    "    return f\"Hello, {name}!\"\n" +
    "if __name__ == \"__main__\":\n" +
    "    print(main(sys.argv[1:]))"
    )
    output = run_python_code_tool.invoke({"file_content": file_content, "argv": ["Areeb"], "timeout_s": 10})
    print(output)
