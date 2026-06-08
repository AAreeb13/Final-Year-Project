from pydantic import BaseModel, Field


class RequirementsSpec(BaseModel):
    functional_requirements: list[str] = Field(default_factory=list)
    non_functional_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    def __str__(self):
        return (
            f"Functional Requirements:\n- " + "\n- ".join(self.functional_requirements) + "\n\n" +
            f"Non-Functional Requirements:\n- " + "\n- ".join(self.non_functional_requirements) + "\n\n" +
            f"Constraints:\n- " + "\n- ".join(self.constraints) + "\n\n" +
            f"Assumptions:\n- " + "\n- ".join(self.assumptions) + "\n\n" +
            f"Out of Scope:\n- " + "\n- ".join(self.out_of_scope)
        )

class QuestionAnswer(BaseModel):
    question: str
    answer: str
    def __str__(self):        return f"Q: {self.question}\nA: {self.answer}"


class RequirementsExtractorInput(BaseModel):
    project_prompt: str
    previous_requirements: RequirementsSpec | None = None
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    def __str__(self):        return (
            f"Project Prompt: {self.project_prompt}\n\n" +
            f"Previous Requirements: \n{self.previous_requirements}\n\n" +
            f"Question-Answer Context: {self.question_answer_context}"
        )


class RequirementsExtractorOutput(BaseModel):
    requirements: RequirementsSpec
    notes: list[str] = Field(default_factory=list)
    def __str__(self):        
        return (
            f"Extracted Requirements: \n{self.requirements}\n\n" +
            f"Extractor Notes: \n- " + "\n- ".join(self.notes)
        )


class RequirementsCriticInput(BaseModel):
    project_prompt: str
    requirements: RequirementsSpec
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    def __str__(self):        
        return (
            f"Project Prompt: {self.project_prompt}\n\n" +
            f"Requirements to Critique: \n{self.requirements}\n\n" +
            f"Question-Answer Context: {self.question_answer_context}"
    )


class RequirementsCriticOutput(BaseModel):
    approved: bool = False
    verdict: str = ""
    questions: list[str] = Field(default_factory=list)
    feedback: str = ""
    def __str__(self):
        return (
            f"Approved: {self.approved}\n\n" +
            f"Verdict: {self.verdict}\n\n" +
            f"Questions from Critic: \n- " + "\n- ".join(self.questions) + "\n\n" +
            f"Critic Feedback: {self.feedback}"
        )

class RequirementsStageOutput(BaseModel):
    requirements: RequirementsSpec
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    critic_verdict: str
    approved: bool
    def __str__(self):
        return (
            f"Requirements: \n{self.requirements}\n\n" +
            f"Question-Answer Context: \n{self.question_answer_context}\n\n" +
            f"Critic Verdict: {self.critic_verdict}\n\n" +
            f"Approved: {self.approved}"
        )
