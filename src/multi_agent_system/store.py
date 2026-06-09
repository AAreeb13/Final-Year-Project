from __future__ import annotations

import re
from pathlib import Path

from src.multi_agent_system.output_schema import (
    ComponentDecompositionOutput,
    ComponentExtractionOutput,
    EnvironmentSetupExecution,
    EnvironmentSetupPlan,
    EnvironmentSetupStageOutput,
    ModuleDesignOutput,
    ProjectStore,
    RepositoryStructure,
    RequirementsStageOutput,
)
from src.settings import settings


PROJECT_STORE_FILENAME = "project_store.json"
PROJECT_STORE_ROOT = ".multi_agent_system/projects"
PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class ProjectStoreRepository:
    """Deterministic JSON persistence for approved multi-agent artifacts."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        root_value = workspace_root or settings.WORKPLACE_FOLDER
        if root_value is None:
            raise ValueError("WORKPLACE_FOLDER must be set to use ProjectStoreRepository")
        self.workspace_root = Path(root_value).expanduser().resolve()

    def create_project(self, project_id: str, project_prompt: str, overwrite: bool = False) -> ProjectStore:
        store_path = self._store_path(project_id)
        if store_path.exists() and not overwrite:
            return self.load_project(project_id)

        store = ProjectStore(
            project_id=project_id,
            project_prompt=project_prompt,
            stage_statuses={"project": "created"},
        )
        self.save_project(store)
        return store

    def load_project(self, project_id: str) -> ProjectStore:
        store_path = self._store_path(project_id)
        if not store_path.exists():
            raise FileNotFoundError(f"Project store not found: {store_path}")
        return ProjectStore.model_validate_json(store_path.read_text(encoding="utf-8"))

    def save_project(self, store: ProjectStore) -> ProjectStore:
        store_path = self._store_path(store.project_id)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = store_path.with_suffix(".tmp")
        temp_path.write_text(
            store.model_dump_json(indent=2),
            encoding="utf-8",
        )
        temp_path.replace(store_path)
        return store

    def save_requirements(self, project_id: str, output: RequirementsStageOutput) -> ProjectStore:
        if not output.approved:
            raise ValueError("Only approved requirements can be saved to the project store")
        store = self.load_project(project_id)
        store.requirements = output
        store.stage_statuses["requirements"] = "complete"
        return self.save_project(store)

    def save_component_extraction(
        self,
        project_id: str,
        output: ComponentExtractionOutput,
    ) -> ProjectStore:
        store = self.load_project(project_id)
        store.component_extraction = output
        store.stage_statuses["design.extract_components"] = "complete"
        return self.save_project(store)

    def save_component_decomposition(
        self,
        project_id: str,
        component_name: str,
        output: ComponentDecompositionOutput,
    ) -> ProjectStore:
        if not component_name:
            raise ValueError("component_name is required")
        store = self.load_project(project_id)
        store.component_decompositions[component_name] = output
        store.stage_statuses[f"design.decompose_component.{component_name}"] = "complete"
        return self.save_project(store)

    def save_module_design(
        self,
        project_id: str,
        component_name: str,
        output: ModuleDesignOutput,
    ) -> ProjectStore:
        if not component_name:
            raise ValueError("component_name is required")
        store = self.load_project(project_id)
        store.module_designs[component_name] = output
        store.stage_statuses[f"design.design_modules.{component_name}"] = "complete"
        return self.save_project(store)

    def save_repository_structure(
        self,
        project_id: str,
        output: RepositoryStructure,
    ) -> ProjectStore:
        store = self.load_project(project_id)
        store.repository_structure = output
        store.stage_statuses["environment.repository_structure"] = "approved"
        return self.save_project(store)

    def save_environment_setup_plan(
        self,
        project_id: str,
        output: EnvironmentSetupStageOutput,
    ) -> ProjectStore:
        if not output.approved:
            raise ValueError("Only approved environment setup plans can be saved to the project store")
        store = self.load_project(project_id)
        store.repository_structure = output.repository_structure
        store.environment_setup = output.environment_setup
        store.stage_statuses["environment.project_setup"] = "approved"
        return self.save_project(store)

    def save_environment_setup_execution(
        self,
        project_id: str,
        output: EnvironmentSetupExecution,
    ) -> ProjectStore:
        store = self.load_project(project_id)
        store.environment_setup_execution = output
        store.stage_statuses["environment.project_setup"] = output.status
        return self.save_project(store)

    def update_stage_status(self, project_id: str, stage: str, status: str) -> ProjectStore:
        store = self.load_project(project_id)
        store.stage_statuses[stage] = status
        return self.save_project(store)

    def _store_path(self, project_id: str) -> Path:
        self._validate_project_id(project_id)
        store_path = (
            self.workspace_root
            / PROJECT_STORE_ROOT
            / project_id
            / PROJECT_STORE_FILENAME
        ).resolve()
        if self.workspace_root != store_path and self.workspace_root not in store_path.parents:
            raise ValueError("Project store path must stay inside the workspace root")
        return store_path

    @staticmethod
    def _validate_project_id(project_id: str) -> None:
        if not project_id:
            raise ValueError("project_id is required")
        if not PROJECT_ID_PATTERN.fullmatch(project_id):
            raise ValueError(
                "project_id may only contain letters, numbers, underscores, hyphens, and periods"
            )


def create_or_load_project(project_id: str, project_prompt: str) -> ProjectStore:
    return ProjectStoreRepository().create_project(project_id, project_prompt)
