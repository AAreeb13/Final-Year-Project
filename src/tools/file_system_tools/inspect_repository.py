"""Repository inspection tools."""

from pathlib import Path
from typing import Iterable

from langchain_core.tools import tool

from src.settings import settings


DEFAULT_IGNORED_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "node_modules",
}


def _resolve_inside_workspace(relative_path: str | None = None) -> Path:
    workspace_root = Path(settings.WORKPLACE_FOLDER).expanduser().resolve()

    if relative_path in (None, "", "."):
        target_path = workspace_root
    else:
        target_path = (workspace_root / relative_path).expanduser().resolve()

    if target_path != workspace_root and workspace_root not in target_path.parents:
        raise ValueError("Path must stay inside the configured workplace folder.")

    if not target_path.exists():
        raise FileNotFoundError(f"Path does not exist: {target_path}")

    return target_path


def _normalise_ignored_names(extra_ignored_names: Iterable[str] | None) -> set[str]:
    ignored_names = set(DEFAULT_IGNORED_NAMES)
    if extra_ignored_names is not None:
        ignored_names.update(name for name in extra_ignored_names if name)

    return ignored_names


def _build_tree_lines(
    root: Path,
    ignored_names: set[str],
    max_depth: int,
    max_entries: int,
) -> tuple[list[str], bool]:
    lines = [f"{root.name}/"]
    entries_seen = 0
    truncated = False

    def add_children(directory: Path, prefix: str, depth: int) -> None:
        nonlocal entries_seen, truncated

        if truncated or depth >= max_depth:
            return

        children = sorted(
            (
                child
                for child in directory.iterdir()
                if child.name not in ignored_names
            ),
            key=lambda child: (not child.is_dir(), child.name.lower()),
        )

        for index, child in enumerate(children):
            if entries_seen >= max_entries:
                truncated = True
                return

            entries_seen += 1
            is_last = index == len(children) - 1
            connector = "`-- " if is_last else "|-- "
            child_prefix = "    " if is_last else "|   "
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{prefix}{connector}{child.name}{suffix}")

            if child.is_dir():
                add_children(child, prefix + child_prefix, depth + 1)

    add_children(root, "", 0)
    return lines, truncated


@tool
def inspect_repository_structure(
    relative_path: str | None = None,
    max_depth: int = 4,
    max_entries: int = 200,
    extra_ignored_names: list[str] | None = None,
) -> str:
    """
    Return a tree view of the configured workplace repository.

    Args:
        relative_path: Optional path inside the workplace folder to inspect.
        max_depth: Maximum directory depth to include.
        max_entries: Maximum number of files/directories to return.
        extra_ignored_names: Additional file or directory names to omit.
    """
    target_path = _resolve_inside_workspace(relative_path)
    ignored_names = _normalise_ignored_names(extra_ignored_names)
    bounded_max_depth = max(1, min(max_depth, 10))
    bounded_max_entries = max(1, min(max_entries, 1000))

    if target_path.is_file():
        return str(target_path.relative_to(Path(settings.WORKPLACE_FOLDER).resolve()))

    lines, truncated = _build_tree_lines(
        target_path,
        ignored_names,
        bounded_max_depth,
        bounded_max_entries,
    )

    if truncated:
        lines.append(f"... truncated after {bounded_max_entries} entries")

    return "\n".join(lines)


if __name__ == "__main__":
    print(inspect_repository_structure.invoke({}))
