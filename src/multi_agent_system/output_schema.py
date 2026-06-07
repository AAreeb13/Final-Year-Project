from pydantic import BaseModel


class RequirementsSpec(BaseModel):
    functional_requirements: list[str] = []
    non_functional_requirements: list[str] = []
    constraints: list[str] = []
    assumptions: list[str] = []
    out_of_scope: list[str] = []

class QuestionAnswer(BaseModel):
    question: str
    answer: str


class RequirementsExtractorInput(BaseModel):
    project_prompt: str
    previous_requirements: RequirementsSpec | None = None
    question_answer_context: list[QuestionAnswer] = []


class RequirementsExtractorOutput(BaseModel):
    requirements: RequirementsSpec
    notes: list[str] = []


class RequirementsCriticInput(BaseModel):
    project_prompt: str
    requirements: RequirementsSpec
    question_answer_context: list[QuestionAnswer] = []

class RequirementsStageOutput(BaseModel):
    requirements: RequirementsSpec
    question_answer_context: list[QuestionAnswer] = []
    critic_verdict: str
    approved: bool
