from pathlib import Path
from typing import TypedDict

class GitCLIError(Exception):
    pass

class GitOutput(TypedDict):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    success: bool


class GitCLI:

    def __init__(self, repo_path):
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.is_dir():
            raise GitCLIError(f"Provided path '{repo_path}' is not a directory.")

        if not (self.repo_path / ".git").exists():
            raise GitCLIError(f"Provided path '{repo_path}' is not a git repository.")
        
        # check if the repo_path is inside the workplace folder
        from settings import settings
        workplace_folder = Path(settings.WORKPLACE_FOLDER).resolve()
        if not self.repo_path.is_relative_to(workplace_folder):
            raise GitCLIError(f"Provided repository path '{repo_path}' is not inside the workplace folder '{workplace_folder}'.")
    

    def _run_git_command(self, args: list[str], timeout: int = 30) -> GitOutput:
        import subprocess

        try: 
            result = subprocess.run(
                ["git", *args],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return GitOutput(
                command=" ".join(["git"] + args),
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.returncode == 0
            )
        except subprocess.TimeoutExpired as e:
            raise GitCLIError(f"Git command timed out: {' '.join(['git'] + args)}") from e

            # return GitOutput(command=" ".join(["git"] + args), exit_code=-1, stdout="",
            #     stderr=f"Git command timed out after {timeout} seconds.", success=False
            # )
        except Exception as e:
            raise GitCLIError(f"Error running git command: {' '.join(['git'] + args)}") from e
            # return GitOutput(command=" ".join(["git"] + args), exit_code=-1, stdout="", 
            #      stderr=f"Unexpected error occurred.{str(e)}", success=False
            # )

    def status(self) -> GitOutput:
        return self._run_git_command(["status"])
    
    def diff(self, *args: str) -> GitOutput:
        return self._run_git_command(["diff", *args]
    )

    def current_branch(self) -> GitOutput:
        return self._run_git_command(["branch", "--show-current"])
    
    def log(self, limit: int = 10, oneline: bool = False) -> GitOutput:
        args = ["log", f"--max-count={limit}"]
        if oneline:
            args.append("--oneline")
        return self._run_git_command(args)

    def stage_items(self, *items: str) -> GitOutput:
        return self._run_git_command(["add", *items])
    

if __name__ == "__main__":
    repo_path = input("Enter the path to the git repository: ").strip()
    try:
        git_cli = GitCLI(repo_path)
        print(f"Successfully initialized GitCLI for repository at '{repo_path}'")
    except GitCLIError as e:
        print(f"Error initializing GitCLI: {e}")
