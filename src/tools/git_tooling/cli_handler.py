import subprocess
from dataclasses import dataclass
from pathlib import Path

class GitCLIError(Exception):
    pass

@dataclass
class GitOutput:
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
        from src.settings import settings
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
    
    def create_branch(self, branch_name: str) -> GitOutput:
        # check if the branch already exists
        branch_result = self._run_git_command(["branch", "--list", branch_name])
        if not branch_result.success:
            return self._run_git_command(["branch", branch_name])
    def log(self, limit: int = 10, oneline: bool = True) -> GitOutput:
        args = ["log", f"--max-count={limit}"]
        if oneline:
            args.append("--oneline")
        return self._run_git_command(args)

    def stage_and_commit_items(self, *items: str, message: str ) -> GitOutput:
        stage_result = self._run_git_command(["add", *items])
        if not stage_result.success:
            return stage_result
        return self._run_git_command(["commit", "-m", message])
    
    def push(self, remote: str = "origin", branch: str | None = None) -> GitOutput:
        if branch is None:
            branch_result = self.current_branch()
            if not branch_result.success:
                return branch_result
            branch = branch_result.stdout.strip()
        return self._run_git_command(["push", remote, branch])
    
    @staticmethod
    def clone_repository(repo_url: str) -> GitOutput:
        from src.settings import settings

        workspace_path = settings.WORKPLACE_FOLDER

        # check if the repo_url is a GitHub URL and ends with .git and if a GitHub PAT is available in settings
        if repo_url.startswith("https://github.com/") and repo_url.endswith(".git") and settings.GITHUB_PERSONAL_ACCESS_TOKEN:
            github_pat = settings.GITHUB_PERSONAL_ACCESS_TOKEN
        else:
            github_pat = None
            raise GitCLIError("Only GitHub repositories are supported for cloning at this time.")

        if github_pat:
            # Prepend the GitHub PAT to the repository URL
            repo_url = repo_url.replace("https://github.com/", f"https://{github_pat}@github.com/")
        
        args = ["clone", repo_url]

        try:
            # Run the git clone command in the workplace folder
            result = subprocess.run(
                ["git", *args],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            return GitOutput(
                command=" ".join(["git"] + args),
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.returncode == 0
            )
        except subprocess.TimeoutExpired as e:
            raise GitCLIError(f"Git clone command timed out: {' '.join(['git'] + args)}") from e
        except Exception as e:
            raise GitCLIError(f"Error running git clone command: {' '.join(['git'] + args)}") from e

if __name__ == "__main__":
    repo_path = input("Enter the path to the git repository: ").strip()
    try:
        git_cli = GitCLI(repo_path)
        print(f"Successfully initialized GitCLI for repository at '{repo_path}'")
    except GitCLIError as e:
        print(f"Error initializing GitCLI: {e}")


    # test clone 

    repo_url = input("Enter the GitHub repository URL to clone (or leave blank to skip cloning): ").strip()

    if repo_url:
        try:
            clone_result = GitCLI.clone_repository(repo_url)
            print(clone_result)
            if clone_result.success:
                print(f"Successfully cloned repository: {clone_result.stdout}")
            else:
                print(f"Failed to clone repository: {clone_result.stderr}")
        except GitCLIError as e:
            print(f"Error cloning repository: {e}")

