
# Requirements Extractor Agent
# Schema:
# Input:
# Natural Language Prompt
# {  "project_id": "todo_list_cli",
#    "project_prompt": "Design a command-line to-do list application that allows users to}

# Output:
# ExtractorSpec {
#   
#   functional_requirements: list[str],
#   non_functional_requirements: list[str],
#   constraints: list[str],
#   assumptions: list[str],
#   out_of_scope: list[str],
## if speaking to Critic Agent, also include:
#   Question_answer_context?: Dict[Str, Str] Question to Answer, that the critic previously returned as part of their feedback, and the corresponding answer provided by the system.
#   prompt?: str
# This is manually provided by the human in the loop
# 
# }


# Requirements Critic Agent
# Schema:
# Input:
# ExtractorSpec {
#   functional_requirements: list[str],
#   non_functional_requirements: list[str],
#   constraints: list[str],
#   assumptions: list[str],
#   out_of_scope: list[str],
#   Below are now guarrunteed to be in the input.
#   question_answer_context: Dict[Str, Str] Question to Answer, that the critic
#   prompt: str
# }


# There will be a few output schemas for requirement extractor
# 1. Requirements Spec -> Critic 1. Requriements Spec with previous Q and A context ->Critic 1. Requirements Spec -> Orchestrator

