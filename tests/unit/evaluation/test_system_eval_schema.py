from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.system_eval_schema import SystemRunOutput


def test_system_run_output_moves_module_fields_out_of_components():
    response = SystemRunOutput.model_validate(
        {
            "components": [
                {
                    "name": "RequirementExtractor",
                    "component": "Analysis",
                    "responsibilities": ["Extract requirements from user input"],
                    "signatures": [
                        {
                            "type": "function",
                            "name": "extract_requirements",
                            "inputs": [{"name": "prompt", "type": "str"}],
                            "output": "RequirementsSpec",
                            "description": "Extract structured requirements.",
                        }
                    ],
                }
            ]
        }
    )

    assert response.components[0].name == "RequirementExtractor"
    assert not hasattr(response.components[0], "signatures")
    assert response.modules[0].name == "RequirementExtractor"
    assert response.modules[0].component == "Analysis"
    assert response.modules[0].signatures[0].name == "extract_requirements"


def test_system_run_output_accepts_common_model_shorthand():
    response = SystemRunOutput.model_validate(
        {
            "components": [
                {
                    "name": "Game Engine",
                    "responsibilities": ["Manage game state and logic"],
                    "technologies": "JavaScript, HTML5 Canvas",
                }
            ],
            "modules": [
                {
                    "name": "updateGame",
                    "component": "Game Engine",
                    "type": "function",
                    "dependencies": "json",
                    "signatures": [
                        {
                            "type": "function",
                            "name": "updateGame",
                            "inputs": ["deltaTime"],
                            "output": "void",
                            "description": "Updates the game state.",
                        }
                    ],
                }
            ],
        }
    )

    assert response.components[0].technologies == [
        "JavaScript",
        "HTML5 Canvas",
    ]
    assert response.modules[0].signatures[0].inputs[0].name == "deltaTime"
    assert response.modules[0].signatures[0].inputs[0].type == ""
    assert response.modules[0].dependencies == ["json"]


def test_system_run_output_drops_extra_fields_and_normalises_nested_shapes():
    response = SystemRunOutput.model_validate(
        {
            "unexpected_top_level": "ignored",
            "requirements": {
                "functional": "Add item, Delete item",
                "extra": "ignored",
            },
            "high_level_design": {
                "components": ["CLI"],
                "relationships": ["uses"],
                "technologies": "Python",
                "extra": "ignored",
            },
            "components": [
                {
                    "name": "CLI",
                    "responsibilities": "Handle commands",
                    "inputs": "argv",
                    "outputs": "printed output",
                    "dependencies": "todo_list",
                    "technologies": "Python",
                    "extra": "ignored",
                }
            ],
            "repository_structure": {
                "directories": [
                    {
                        "name": "todo_app",
                        "children": "tests",
                        "files": [
                            {
                                "name": "main.py",
                                "modules": "main",
                                "extra": "ignored",
                            }
                        ],
                        "extra": "ignored",
                    }
                ],
                "extra": "ignored",
            },
            "implementation_plan": ["Create CLI"],
            "test_plan": {
                "unit_tests": "test add, test delete",
                "commands": "pytest",
                "extra": "ignored",
            },
            "generated_files": [
                {
                    "path": "todo_app/main.py",
                    "content": "print('ok')",
                    "extra": "ignored",
                }
            ],
            "limitations": "No persistence",
        }
    )

    assert response.requirements.functional == ["Add item", "Delete item"]
    assert response.high_level_design.components[0].name == "CLI"
    assert response.high_level_design.relationships[0].relationship_type == "uses"
    assert response.high_level_design.technologies == ["Python"]
    assert response.components[0].responsibilities == ["Handle commands"]
    assert response.components[0].dependencies == ["todo_list"]
    assert response.repository_structure.directories[0].children == ["tests"]
    assert response.repository_structure.directories[0].files[0].modules == ["main"]
    assert response.implementation_plan[0].action == "Create CLI"
    assert response.test_plan.unit_tests == ["test add", "test delete"]
    assert response.test_plan.commands == ["pytest"]
    assert response.generated_files[0].path == "todo_app/main.py"
    assert response.limitations == ["No persistence"]
