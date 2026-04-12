
import os
import time
from typing import List
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, SecretStr

from src.tools.python_code_execution.tool import run_python_code_tool

def import_settings():
    from src.settings import settings
    return settings

class SingleAgentModelOutput(BaseModel):
    # Task Decomposition
    functional_requirements: List[str]
    # Top-Down Decomposition
    
    components: List[str]
    relationships: List[str]
    # System Design
    solid_principles_used: List[str]
    solid_principles_jeopardised: List[str]
    
    folder_structure_json: str
    # Code Generation 
    filename_to_code: dict[str, str] = dict()  # filename: code
    
if __name__ == "__main__":
    settings = import_settings()

    assert settings.OPEN_ROUTER_KEY is not None, "openrouterkey must be set in the .env file"
    print("Settings imported successfully. openrouterkey is set.")
    model = ChatOpenAI(
        api_key=SecretStr(settings.OPEN_ROUTER_KEY),
        base_url="https://openrouter.ai/api/v1",
        model="gpt-4o-mini",
        temperature=0.1,
        max_completion_tokens=100000)
    
    
    
    agent = create_agent(
        model=model,
        system_prompt=("You are multi-purpose agent that aims to follow Software Development LifeCycle. You must perform the following tasks in order\n" + 
        "1. Perform Task Decomposition: Decompose the problem and extract functional requirements from a problem\n. Do not include any requirements that were not explicitly suggested." 
        "\n  Incorrect Example: Input=\"Iris Detector\" Functional Requirement: \"Authentication, user registration\"\n" +
        "2. Perform Top-Down Decomposition to highlight high-level components and relationships between components\n" +
        "3. System Design: Using the extracted components and relationships, perform high-level system design that describes which components will be functions, classes (super-classes, abstract classes) and APIs. Be sure to identify parameters."
        "   You must adhere to SOLID principles and identify where SOLID principles are being adhered to. \n"
        "4. Provide a clear project folder structure\n"
        "5. Code Generation: write Python code, for each file with comments of which SOLID principle is used and which principles may be jeopardised. Complete all the code and do not leave any sections empty\n"
        "Tool contract for run_python_code_tool: pass `file_content` (string), optional `argv` (list of CLI args), and optional `timeout_s` (seconds). Do not use stdin JSON parameters.\n"
        "Imports are allowed when required by the task."),
        tools=[run_python_code_tool],
        response_format=SingleAgentModelOutput
    )
    # print("here")
    response_dict = agent.invoke({"messages": [HumanMessage("snake game using pygame")]})
    response: SingleAgentModelOutput = SingleAgentModelOutput.model_validate_json(response_dict["messages"][-1].content)
    # print(response_dict["messages"][-1].content)
    
    # display response in a readable format
    print("Functional Requirements:")
    for req in response.functional_requirements:
        print(f"- {req}")
    print("\nComponents:")
    for comp in response.components:
        print(f"- {comp}")
    print("\nRelationships:")
    for rel in response.relationships:
        print(f"- {rel}")
    print("\nSOLID Principles Used:")
    for solid in response.solid_principles_used:
        print(f"- {solid}")
    print("\nSOLID Principles Jeopardised:")
    for solid in response.solid_principles_jeopardised:
        print(f"- {solid}")
    print("\nFolder Structure:")
    print(response.folder_structure_json)
    print("\nGenerated Code:")
    for filename, code in response.filename_to_code.items():
        print(f"Filename: {filename}\nCode:\n{code}\n{'-'*40}")
    
    # create folder to store response
    current_directory = os.getcwd()
    response_directory = os.path.join(current_directory, ".response")
    os.makedirs(response_directory, exist_ok=True)
    # save response to a text file
    cur_time = int(time.time())
    # convert cur_time to hexadecimal
    cur = hex(cur_time)[2:]
    with open(os.path.join(response_directory, f"response{cur}.json"), "w") as f:      
        print("Saving response to file at location:", os.path.join(response_directory, f"response{cur}.json"))  
        f.write(response_dict["messages"][-1].content)
        

        
 
