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
