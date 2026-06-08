from src.multi_agent_system.graphs.requirements import build_requirements_graph
from src.multi_agent_system.output_schema import QuestionAnswer

graph = build_requirements_graph()

prompt = input("Enter project prompt: ")
current = graph.run(prompt if prompt else "Build a CLI todo app")

while current.status != "complete":
    print("Current status:", current.status)
    print("=========Current State=========\n", current.requirements)
    # print(current.)

    print("\nThe system is waiting for user input to address the critic's feedback.")

    for q in current.questions:
        ans = input(f"Question from critic: {q}\nYour answer: ")
        current.question_answer_context.append(QuestionAnswer(question=q, answer=ans))

    
    current = graph.run(
        prompt if prompt else "Build a CLI todo app",
        previous_requirements=current.requirements,
        question_answer_context=current.question_answer_context,
    )

print("\n=======Final State======\n", current)
