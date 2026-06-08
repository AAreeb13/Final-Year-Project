from __future__ import annotations

from src.multi_agent_system.graphs.design import (
    DesignRunResult,
    UserApprovalDecision,
    build_design_graph,
)
from src.multi_agent_system.graphs.requirements import RequirementsRunResult, build_requirements_graph
from src.multi_agent_system.output_schema import (
    DesignStageOutput,
    DesignTask,
    QuestionAnswer,
    RequirementsStageOutput,
)
from src.multi_agent_system.store import ProjectStoreRepository


DEFAULT_PROMPT = "Build a CLI todo app"


def run_requirements_loop(project_prompt: str) -> RequirementsRunResult:
    graph = build_requirements_graph()
    current = graph.run(project_prompt)

    while current.status != "complete":
        if current.status == "failed":
            raise RuntimeError(
                f"Requirements stage failed: {current.failure_reason or current.critic_output}"
            )
        print("Current requirements status:", current.status)
        print("=========Current Requirements=========\n", current.requirements)
        _append_answers(current.question_answer_context, current.questions)
        current = graph.run(
            project_prompt,
            previous_requirements=current.requirements,
            question_answer_context=current.question_answer_context,
        )

    return current


def run_design_loop(
    project_prompt: str,
    requirements: RequirementsStageOutput,
    task: DesignTask,
    previous_design_output: DesignStageOutput | None = None,
    max_revisions: int = 3,
    auto_approval: bool = False,
) -> DesignRunResult:
    graph = build_design_graph(
        approval_callback=_interactive_design_approval,
        auto_approval=auto_approval,
    )
    current = graph.run(
        project_prompt=project_prompt,
        requirements=requirements,
        task=task,
        previous_design_output=previous_design_output,
    )

    revision_count = 0
    while current.status != "complete":
        if current.status == "failed":
            raise RuntimeError(f"Design stage failed for {task.kind}: {current.critic_output}")
        revision_count += 1
        if revision_count > max_revisions:
            print(
                f"Warning: design revisions exceeded {max_revisions} for {task.kind}. "
                "Continuing with the current design draft."
            )
            return _force_complete_design_result(current)
        print(f"Current design status for {task.kind}:", current.status)
        print("=========Current Design Draft=========\n", current.architect_output)
        feedback = _build_design_revision_feedback(current)
        current = graph.run(
            project_prompt=project_prompt,
            requirements=requirements,
            task=task,
            previous_design_output=current.stage_output or previous_design_output,
            question_answer_context=current.question_answer_context,
            critic_feedback=feedback,
        )

    return current


def _build_design_revision_feedback(current: DesignRunResult) -> str:
    if current.critic_output is None:
        return ""
    feedback_parts = []
    if current.critic_output.feedback:
        feedback_parts.append(current.critic_output.feedback)
    if current.critic_output.required_changes:
        feedback_parts.append(
            "Required changes:\n- " + "\n- ".join(current.critic_output.required_changes)
        )
    if current.critic_output.questions:
        feedback_parts.append(
            "Do not ask the user these design questions. Resolve them with conservative design decisions:\n- "
            + "\n- ".join(current.critic_output.questions)
        )
    return "\n\n".join(feedback_parts)


def _force_complete_design_result(current: DesignRunResult) -> DesignRunResult:
    if current.stage_output is None:
        raise RuntimeError(f"Cannot force-complete {current.task.kind} without a design stage output")
    current.stage_output.approved = True
    current.stage_output.critic_verdict = (
        f"{current.stage_output.critic_verdict}\n\n"
        "Force continued: maximum design revisions exceeded."
    )
    current.status = "complete"
    return current


def _append_answers(context: list[QuestionAnswer], questions: list[str]) -> None:
    print("\nThe system is waiting for user input to address critic feedback.")
    for question in questions:
        answer = input(f"Question from critic: {question}\nYour answer: ")
        context.append(QuestionAnswer(question=question, answer=answer))


def _interactive_design_approval(
    node_name: str,
    step_input: dict,
    step_output: dict,
) -> UserApprovalDecision:
    print(f"\nApproval required for design node: {node_name}")
    print("Output:")
    print(step_output)
    choice = input("Approve this design node output? [Y/n]: ").strip().lower()
    if choice in {"", "y", "yes"}:
        return UserApprovalDecision(approved=True)
    feedback = input("Revision feedback: ").strip()
    return UserApprovalDecision(
        approved=False,
        feedback=feedback,
    )


def main() -> None:
    project_id = input("Enter project id: ").strip() or "todo_cli"
    project_prompt = input("Enter project prompt: ").strip() or DEFAULT_PROMPT
    auto_approve_design = input("Auto-approve design node outputs? [y/N]: ").strip().lower() in {
        "y",
        "yes",
    }

    store_repository = ProjectStoreRepository()
    store = store_repository.create_project(project_id, project_prompt)

    requirements_result = run_requirements_loop(store.project_prompt)
    if requirements_result.stage_output is None:
        raise RuntimeError("Requirements graph completed without a stage output")
    store = store_repository.save_requirements(project_id, requirements_result.stage_output)
    print("\nSaved approved requirements to Project Store.")

    component_result = run_design_loop(
        project_prompt=store.project_prompt,
        requirements=store.requirements,
        task=DesignTask(kind="extract_components"),
        auto_approval=auto_approve_design,
    )
    component_extraction = _require_stage_artifact(component_result).component_extraction
    if component_extraction is None:
        raise RuntimeError("Component extraction completed without component_extraction output")
    store = store_repository.save_component_extraction(project_id, component_extraction)
    print("\nSaved approved component extraction to Project Store.")

    previous_design_output = component_result.stage_output
    for component in component_extraction.components:
        decomposition_result = run_design_loop(
            project_prompt=store.project_prompt,
            requirements=store.requirements,
            task=DesignTask(
                kind="decompose_component",
                target_component=component.name,
            ),
            previous_design_output=previous_design_output,
            auto_approval=auto_approve_design,
        )
        component_decomposition = _require_stage_artifact(decomposition_result).component_decomposition
        if component_decomposition is None:
            raise RuntimeError(f"{component.name} decomposition completed without output")
        store = store_repository.save_component_decomposition(
            project_id,
            component.name,
            component_decomposition,
        )
        print(f"\nSaved approved decomposition for {component.name}.")

        module_result = run_design_loop(
            project_prompt=store.project_prompt,
            requirements=store.requirements,
            task=DesignTask(
                kind="design_modules",
                target_component=component.name,
            ),
            previous_design_output=decomposition_result.stage_output,
            auto_approval=auto_approve_design,
        )
        module_design = _require_stage_artifact(module_result).module_design
        if module_design is None:
            raise RuntimeError(f"{component.name} module design completed without output")
        store = store_repository.save_module_design(
            project_id,
            component.name,
            module_design,
        )
        print(f"\nSaved approved module design for {component.name}.")

    print("\n=======Project Store Updated=======")
    print(store)


def _require_stage_artifact(result: DesignRunResult) -> DesignStageOutput:
    if result.stage_output is None:
        raise RuntimeError(f"{result.task.kind} completed without a stage output")
    return result.stage_output


if __name__ == "__main__":
    main()
