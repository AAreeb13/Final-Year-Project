from pydantic import BaseModel, Field


class RequirementsSpec(BaseModel):
    functional_requirements: list[str] = Field(default_factory=list)
    non_functional_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)

class QuestionAnswer(BaseModel):
    question: str
    answer: str


class RequirementsExtractorInput(BaseModel):
    project_prompt: str
    previous_requirements: RequirementsSpec | None = None
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)


class RequirementsExtractorOutput(BaseModel):
    requirements: RequirementsSpec
    notes: list[str] = Field(default_factory=list)


class RequirementsCriticInput(BaseModel):
    project_prompt: str
    requirements: RequirementsSpec
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)


class RequirementsCriticOutput(BaseModel):
    approved: bool = False
    verdict: str = ""
    questions: list[str] = Field(default_factory=list)
    feedback: str = ""

class RequirementsStageOutput(BaseModel):
    requirements: RequirementsSpec
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    critic_verdict: str
    approved: bool
