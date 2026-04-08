"""
    This is an agent tool that runs python code. It takes in the whole content of the python file and the parameters to run the code with. We will temporarily create a file with the content of the python code and then run the file with the parameters as input. The output of the code will be returned as a string. The file will be deleted after running the code. 
    
    It is important that the file has a __main__ function that takes in the parameters as input and runs the code. The tool will run the code and return the output of the code. 
"""
from langchain.tools import tool
import json
import os
import subprocess
import sys
import tempfile


@tool
def run_python_code_tool(filecontents: str, parameters: dict) -> str:
    """
    Runs the given python code with the given parameters and returns the output as a string. The filecontents should be the whole content of the python file. The parameters should be a dictionary of parameter names and values. The file will be created, run and deleted in a temporary directory. The output of the code will be returned as a string.
    
    Args:
        filecontents (str): The whole content of the python file to be run.
        parameters (dict): A dictionary of parameter names and values to be passed to the code.
    
    Returns:
        str: The output of the code as a string.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(filecontents.encode())
        temp_file_path = temp_file.name
    
    try:
        # Run the python file with JSON-encoded parameters as stdin.
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            input=json.dumps(parameters),
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
        os.remove(temp_file_path)
        

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