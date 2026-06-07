from src.evaluation.helper import build_project_spec_from_yaml

from src.evaluation.system_eval_schema import SystemRunOutput
from src.evaluation.sdlc_eval_schema import ProjectSpec
def compute_basic_metrics(project: ProjectSpec, result: SystemRunOutput):
    return {
    "project_id": project.project_id,
    "system_name": result.system_name,
    "implementation_success": result.status,
    "total_requirements": len(project.requirements.functional) + len(project.requirements.non_functional),
    "implemented_requirements": len(result.requirements.functional) + len(result.requirements.non_functional) if result.requirements else 0,
    "total_components": len(project.components),
    "implemented_components": len(result.components) if result.components else 0,
    "total_tool_calls": len(result.tool_calls) if result.tool_calls else 0,
    "tool_call_success_rate": (sum(1 for call in result.tool_calls if call.success) / len(result.tool_calls))* 100 if result.tool_calls else 0,
}

if __name__ == "__main__":
    system_run_output = SystemRunOutput(
        system_name="ExampleSystem",
        status="success",
        requirements={
            "functional": ["req1", "req2"],
            "non_functional": ["nfr1"]
        },
        components=["comp1", "comp2"],
        tool_calls=[
            {"tool_name": "ToolA", "success": True},
            {"tool_name": "ToolB", "success": False}
        ]
    )
    project_spec = build_project_spec_from_yaml("snake_game")
    metrics = compute_basic_metrics(project_spec, system_run_output)
    print(metrics)
